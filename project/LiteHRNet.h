#pragma once
#include <Windows.h>

const int COCO_N_PARTS = 17;
const int COCO_N_PAIRS = 18;
const int MAX_HUMAN_NUM = 32;

//关键点
struct KeyPoints {
	bool has_value = false; //关键点的是否可用
	float x = 0; // 关键点在输入图像中的坐标
	float y = 0;
	float score = 0; //关键点分数
};
//人体关键点集合
struct BODY {
	KeyPoints parts[COCO_N_PARTS]; //18个关键点
};
//输出结果
struct HUMANS {
	int human_num;  //人数
	BODY bodys[MAX_HUMAN_NUM]; //最多32个人
};

using idx_pair_t = std::pair<int, int>;
using coco_pair_list_t = std::vector<idx_pair_t>;
const coco_pair_list_t COCOPAIRS = {
	{ 1, 2 }, // 6
	{ 1, 5 }, // 10
	{ 2, 3 }, // 7
	{ 3, 4 }, // 8
	{ 5, 6 }, // 11
	{ 6, 7 }, // 12
	{ 1, 8 }, // 0
	{ 8, 9 }, // 1
	{ 9, 10 }, // 2
	{ 1, 11 }, // 3
	{ 11, 12 }, // 4
	{ 12, 13 }, // 5
	{ 1, 0 }, // 14
	{ 0, 14 }, // 15
	{ 14, 16 }, // 17
	{ 0, 15 }, // 16
	{ 15, 17 }, // 18
	{ 2, 16 }, // * 9
	{ 5, 17 }, // * 13
};