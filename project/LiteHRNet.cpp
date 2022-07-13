#include <fstream>
#include <map>
#include <chrono>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "opencv2/opencv.hpp"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 384;
static const int INPUT_W = 288;
static const int OUTPUT_SIZE = 17 * 96 * 72;
//const int batchSize = 64;

const char* INPUT_BLOB_NAME = "image";
const char* OUTPUT_BLOB_NAME = "heatmap";

const float meanVal[3] = { 0.485, 0.456, 0.406 };//RGB
const float stdVal[3] = { 0.229, 0.224, 0.225 };

static const int joint_pairs[16][2] = {
	{0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}
};


using namespace cv;
using namespace std;
using namespace nvinfer1;

static Logger gLogger;

template <class F>
struct _LoopBody : public cv::ParallelLoopBody {
	F f_;
	_LoopBody(F f) : f_(std::move(f)) {}
	void operator()(const cv::Range& range) const override { f_(range); }
};

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) 
{
	const ICudaEngine& engine = context.getEngine();

	// Pointers to input and output device buffers to pass to engine.
	// Engine requires exactly IEngine::getNbBindings() number of buffers.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
	const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// Create GPU buffers on device
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	// Create stream
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.setBindingDimensions(inputIndex, Dims4(batchSize,3, INPUT_H, INPUT_W));
	context.enqueueV2(buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// Release stream and buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}

cv::Mat letterBox(const cv::Mat &src, int net_w, int net_h)
{
	int new_w = src.cols;
	int new_h = src.rows;

	if (((float)net_w / src.cols) < ((float)net_h / src.rows))
	{
		new_w = net_w;
		new_h = (src.rows * net_w) / src.cols;
	}
	else
	{
		new_h = net_h;
		new_w = (src.cols * net_h) / src.rows;
	}

	cv::Mat dest(net_h, net_w, CV_8UC3, cv::Scalar(114, 114, 114));
	cv::Mat embed;
	cv::resize(src, embed, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
	cv::Mat imageROI = dest(cv::Rect((net_w - new_w) / 2, (net_h - new_h) / 2,
		embed.cols, embed.rows));
	embed.copyTo(imageROI);
	return dest;
}

//transform point from input to source image
void scaleCoords(int img1w, int img1h, int img0w, int img0h, cv::Point& p)
{
	float gain = std::min((float)img1h / img0h, (float)img1w / img0w);
	int padw = (img1w - img0w * gain) / 2;
	int padh = (img1h - img0h * gain) / 2;

	p.x = (p.x - padw) / gain;
	p.y = (p.y - padh) / gain;
	//clip
	p.x = std::max(std::min(p.x, img0w), 0);
	p.y = std::max(std::min(p.y, img0h), 0);
}

void normalize(cv::Mat& img,float* data) {
	for (int i = 0; i < img.rows; ++i) {
		uchar* rowData = img.ptr<uchar>(i);
		for (int j = 0; j < img.cols; ++j) {
			data[i* img.cols+j] = ((float)rowData[2] / 255.0 - meanVal[0]) / stdVal[0]; //R
			data[img.rows* img.cols + i * img.cols + j] = ((float)rowData[1] / 255.0 - meanVal[1]) / stdVal[1];
			data[img.rows* img.cols*2 + i * img.cols + j] = ((float)rowData[0] / 255.0 - meanVal[2]) / stdVal[2];
			rowData += 3;
		}
	}
}

/*
get max value and coordinate in single heatmap
heatmap: single batch [1,K,H,W]
shape: [1,K,H,W]
pred: K*3, 3 means x,y,score 
*/
void get_max_pred(float* heatmap, const vector<int>& shape, float* pred) {
	//int batch_size = shape[0];
	int num_joints = shape[1];
	int H = shape[2];
	int W = shape[3];
	cv::parallel_for_(cv::Range(0, num_joints), _LoopBody<std::function<void(const cv::Range&)>>( [&](const cv::Range& r) {
						for (int i = r.start; i < r.end; i++) {
						  float* src_data = heatmap+ i * H * W;
						  cv::Mat mat = cv::Mat(H, W, CV_32FC1, src_data);
						  double min_val, max_val;
						  cv::Point min_loc, max_loc;
						  cv::minMaxLoc(mat, &min_val, &max_val, &min_loc, &max_loc);
						  float* dst_data = pred + i * 3;
						  *(dst_data + 0) = -1;
						  *(dst_data + 1) = -1;
						  *(dst_data + 2) = max_val;
						  if (max_val > 0.0) {
							*(dst_data + 0) = max_loc.x;
							*(dst_data + 1) = max_loc.y;
						  }
						}
					  } ));
}

/*
transform coordinate of max value in heatmap to input image
pred: K*3, 3 means x,y,score 
K: point index
center: bbox center coordinate
_scale: bbox scale w,h
output_size: heatmap size w,h
*/
void transform_pred(float* pred,int k, const vector<float>& center, const vector<float>& _scale, const vector<int>& output_size) {
	auto scale = _scale;
	//scale[0] *= 200; 
	//scale[1] *= 200;

	float scale_x, scale_y;
	scale_x = scale[0] / output_size[0];
	scale_y = scale[1] / output_size[1];

	float* data = pred + k * 3;
	*(data + 0) = *(data + 0) * scale_x + center[0] - scale[0] * 0.5;
	*(data + 1) = *(data + 1) * scale_y + center[1] - scale[1] * 0.5;
}

/*
transform single heatmap to keypoints in image
heatmap: single batch [1,K,H,W]
shape: [1,K,H,W]
center: bbox center coordinate
_scale: bbox scale h,w
pred: K*3, 3 means x,y,score
*/
void keypoints_from_heatmap(float* heatmap,const vector<int>& shape, const vector<float>& center, const vector<float>& scale, float* pred) {
	int K = shape[1];
	int H = shape[2];
	int W = shape[3];
	get_max_pred(heatmap, shape, pred);
	for (int i = 0; i < K; i++) {
		float* data = heatmap + i * W * H;
		auto _data = [&](int y, int x) { return *(data + y * W + x); };
		int px = *(pred + i * 3 + 0);
		int py = *(pred + i * 3 + 1);
		if (1 < px && px < W - 1 && 1 < py && py < H - 1) {
			float v1 = _data(py, px + 1) - _data(py, px - 1);
			float v2 = _data(py + 1, px) - _data(py - 1, px);
			//shift a little for higher acc
			*(pred + i * 3 + 0) += (v1 > 0) ? 0.25 : ((v1 < 0) ? -0.25 : 0);
			*(pred + i * 3 + 1) += (v2 > 0) ? 0.25 : ((v2 < 0) ? -0.25 : 0);
		}
	}
	// Transform back to the image
	for (int i = 0; i < K; i++) {
		transform_pred(pred,i, center, scale, { W,H });
	}
}

int main()
{
	string engineFile = "lite-hrnet//litehrnet_30_coco_384x288-dynamic.trt";

	std::ifstream file(engineFile, std::ios::binary);
	if (!file.good()) {
		std::cerr << "read " << engineFile << " error!" << std::endl;
		return 0;
	}
	char *trtModelStream = nullptr;
	size_t size = 0;
	file.seekg(0, file.end);
	size = file.tellg();
	file.seekg(0, file.beg);
	trtModelStream = new char[size];
	assert(trtModelStream);
	file.read(trtModelStream, size);
	file.close();

	IRuntime* runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
	assert(engine != nullptr);
	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);
	if (trtModelStream != nullptr)
		delete[] trtModelStream;

	vector<Mat> images;
	Mat srcImg1 = imread("1.jpg");
	Mat srcImg2 = imread("2.jpg");
	Mat srcImg3 = imread("3.jpg");
	images.push_back(srcImg1);
	images.push_back(srcImg2);
	images.push_back(srcImg3);
	for (int i = 0; i < 61; i++)
	{
		Mat srcImgi = imread("2.jpg");
		images.push_back(srcImgi);
	}
	
	int batchSize = images.size();
	int singleInputSize = 3 * INPUT_W*INPUT_H;
	float * batchInput= new float[batchSize*singleInputSize];
	for (int i = 0; i < batchSize; i++)
	{
		Mat inputImg = letterBox(images[i], INPUT_W, INPUT_H);
		normalize(inputImg, batchInput + i * singleInputSize);
	}

	float* batchOutput=new float[batchSize*OUTPUT_SIZE];
	doInference(*context, batchInput, batchOutput, batchSize);//warmup
	auto start = std::chrono::high_resolution_clock::now();

	doInference(*context, batchInput, batchOutput, batchSize);

	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "batch size: " << batchSize<<", infer time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" <<std::endl;

	float* pred = new float[17 * 3];
	vector<int> shape = { 1,17,96,72 };
	vector<float> center = { (float)INPUT_W / 2,(float)INPUT_H / 2 };
	vector<float> scale = { (float)INPUT_W,(float)INPUT_H };
	for (int i = 0; i < batchSize; i++)
	{
		keypoints_from_heatmap(batchOutput +i* OUTPUT_SIZE, shape, center, scale, pred);

		vector<Point> keyPoints;
		for (int j = 0; j < 17; j++)
		{
			cv::Point p = cv::Point(*(pred + j * 3), *(pred + j * 3 + 1));
			scaleCoords(INPUT_W, INPUT_H, images[i].cols, images[i].rows, p);
			keyPoints.push_back(p);
			cv::circle(images[i], p, 3, cv::Scalar(0, 255, 0),3);
		}
		for (int j = 0; j < 16; j++)
		{
			cv::line(images[i], keyPoints[joint_pairs[j][0]], keyPoints[joint_pairs[j][1]], cv::Scalar(255, 0, 255),2);
		}
		string outName = "images/out"+to_string(i) + ".jpg";
		imwrite(outName, images[i]);
	}
	delete[] pred;
	delete[] batchInput;
	delete[] batchOutput;
	// Destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	return 0;
}
