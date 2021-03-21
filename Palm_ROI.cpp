//#include<opencv2\core\core.hpp>
#include<opencv2\opencv.hpp>
#include<opencv2\ml.hpp>
#include<iostream>
#include<stdio.h>
#include<math.h>
#include <string.h>
#include<vector>
#include<time.h> 
//#include <omp.h>
//#include <windows.h>
#include <io.h>
#include <direct.h>

#define L_BASE 100
#define HIGH 255
#define LOW 0
#define TMP 128
#define PI 3.14159265359


using namespace cv;
using namespace std;
using namespace cv::ml;

Mat catch_palm_ROI(Mat &a_timage_in, Mat a_timage_plam, Mat &strong_image, string hand, string name);/*輸入乾淨的二質化圖片，並回傳ROI圖片，in=二質化圖片,plam=手掌原圖,strong_img可將圖片回傳主程式的暫存器*/

void labelset(Mat a_timage_in, int xs, int ys, int label)
{
	int	i, j, cnt, im, ip, jm, jp;

	a_timage_in.at<uchar>(ys, xs) = label;
	for (;;) {
		cnt = 0;
		for (i = 0; i < a_timage_in.rows; i++)
			for (j = 0; j < a_timage_in.cols; j++)
				if (a_timage_in.at<uchar>(i, j) == label) {
					im = i - 1; ip = i + 1; jm = j - 1; jp = j + 1;
					if (im < 0) im = 0; if (ip >= a_timage_in.rows) ip = a_timage_in.rows - 1;
					if (jm < 0) jm = 0; if (jp >= a_timage_in.cols) jp = a_timage_in.cols - 1;
					if (a_timage_in.at<uchar>(i, jp) == HIGH) {
						a_timage_in.at<uchar>(i, jp) = label; cnt++;
					}
					if (a_timage_in.at<uchar>(im, jp) == HIGH) {
						a_timage_in.at<uchar>(im, jp) = label; cnt++;
					}
					if (a_timage_in.at<uchar>(im, j) == HIGH) {
						a_timage_in.at<uchar>(im, j) = label; cnt++;
					}
					if (a_timage_in.at<uchar>(im, jm) == HIGH) {
						a_timage_in.at<uchar>(im, jm) = label; cnt++;
					}
					if (a_timage_in.at<uchar>(i, jm) == HIGH) {
						a_timage_in.at<uchar>(i, jm) = label; cnt++;
					}
					if (a_timage_in.at<uchar>(ip, jm) == HIGH) {
						a_timage_in.at<uchar>(ip, jm) = label; cnt++;
					}
					if (a_timage_in.at<uchar>(ip, j) == HIGH) {
						a_timage_in.at<uchar>(ip, j) = label; cnt++;
					}
					if (a_timage_in.at<uchar>(ip, jp) == HIGH) {
						a_timage_in.at<uchar>(ip, jp) = label; cnt++;
					}
				}
		if (cnt == 0) break;
	}
}
int labeling(Mat a_timage_in, Mat &imagelabel)
{
	int	i, j, label;

	for (i = 0; i < a_timage_in.rows; i++)
		for (j = 0; j < a_timage_in.cols; j++)
			a_timage_in.at<uchar>(i, j) = 255 - a_timage_in.at<uchar>(i, j);
	imagelabel = a_timage_in.clone();
	label = L_BASE;
	for (i = 0; i < a_timage_in.rows; i++)
		for (j = 0; j < a_timage_in.cols; j++) {
			if (imagelabel.at<uchar>(i, j) == HIGH) {
				if (label >= HIGH) return -1;
				labelset(imagelabel, j, i, label); label++;
			}
		}
	return label - L_BASE;
}
Mat labeling_replace(Mat a_timage_in)
{
	Mat a_timage_out = a_timage_in.clone();
	int label = L_BASE;
	int number_counter = 1;
	int total_pixel = a_timage_in.rows*a_timage_in.cols;
	while (number_counter != 0)
	{
		for (int i = 0; i < a_timage_in.rows; i++)
		{
			for (int j = 0; j < a_timage_in.cols; j++)
			{
				if (a_timage_in.at<uchar>(i, j) == label)	number_counter++;
			}
		}
		if (number_counter < total_pixel / 100 * 3)
		{
			for (int i = 0; i < a_timage_in.rows; i++)
			{
				for (int j = 0; j < a_timage_in.cols; j++)
				{
					if (a_timage_in.at<uchar>(i, j) == label)
					{
						a_timage_out.at<uchar>(i, j) = 0;
					}
				}
			}
		}
		else if (number_counter>total_pixel / 100 * 3)
		{
			for (int i = 0; i < a_timage_in.rows; i++)
			{
				for (int j = 0; j < a_timage_in.cols; j++)
				{
					if (a_timage_in.at<uchar>(i, j) == label)
					{
						a_timage_out.at<uchar>(i, j) = 255;
					}
				}
			}
		}
		label++;
		if (number_counter != 1)
		{
			number_counter = 1;
		}
		else if (number_counter == 1)
		{
			number_counter = 0;
		}
	}
	return a_timage_out;
}


Mat transpose(Mat g_utimage_in)
{
	Mat g_utimage_out(g_utimage_in.cols, g_utimage_in.rows, g_utimage_in.type());
	for (int i = 0; i < g_utimage_in.rows; i++)
	{
		for (int j = 0; j < g_utimage_in.cols; j++)
		{
			g_utimage_out.at<uchar>(j, i) = g_utimage_in.at<uchar>(i, j);
		}
	}
	flip(g_utimage_out, g_utimage_out, 0);
	return g_utimage_out;
}


int main()
{
/*	string user_num = "0";
	String log_path = "C:\\Deep_learning\\dataset\\Palm-image\\ROI_log_file_" + user_num + "\\";
	String store_path = "C:\\Deep_learning\\dataset\\Palm-image\\Palm_ROI_" + user_num + "\\";
	String mask_path = "C:\\Deep_learning\\dataset\\Palm-image\\predict_"+user_num+"\\";
	String original_path = "C:\\Deep_learning\\dataset\\Palm-image\\data_val_"+user_num+"\\";*/
	/*String log_path = "C:\\Users\\raychen\\Desktop\\ROI\\";
	String store_path = "C:\\Users\\raychen\\Desktop\\ROI\\" ;
	String mask_path = "C:\\Image\\Plam_CASIA\\Image\\850nm\\mask\\";
	String original_path = "C:\\Image\\Plam_CASIA\\Image\\850nm\\original\\";*/
//	log_path = "D:\\Palm_dataset\\HNCB\\log\\2000\\";
//	store_path = "D:\\Palm_dataset\\HNCB\\ROI\\2000\\";
//	mask_path = "D:\\Palm_dataset\\HNCB\\mask\\2000\\";
//	original_path = "D:\\Palm_dataset\\HNCB\\2000";
	

	string temp_log = "C:\\Image\\ours_database\\Log\\";
	string temp_store = "C:\\Image\\ours_database\\ROI\\";
	string temp_mask = "C:\\Image\\ours_database\\Mask\\";
	string temp_original = "C:\\Image\\ours_database\\original\\";
	//#pragma omp parallel for num_threads(3)
//#pragma omp parallel for
	for (int user_num = 37; user_num < 41 ; user_num++)
	{
		stringstream ss;
		ss << setw(3) << setfill('0') << user_num;
		string string_user;
		ss >> string_user;
		String log_path = temp_log + string_user + "\\";
		String store_path = temp_store + string_user + "\\";
		String mask_path = temp_mask + string_user + "\\";
		String original_path = temp_original + string_user + "\\";
		if (_access(original_path.c_str(),06) == -1)
			continue;
		if (_access(store_path.c_str(), 06) == -1)
			_mkdir(store_path.c_str());
		if (_access(log_path.c_str(), 06) == -1)
			_mkdir(log_path.c_str());

		//vector<String> mask;
		//glob(mask_path, mask, false);
		vector<String> original;
		glob(original_path, original, false);
		//#pragma omp parallel for
		fstream  fp;  //宣 告一個fstream物 件
		fp.open("error_log.txt", ios::in);
		if (!fp) //如果開啟檔案失敗，fp為0；成功，fp為非0     
		{
			cout << "Fail to open error_log " << endl;
		}
		string inputStr;
		vector<string> inputContent;
		while (getline(fp, inputStr)) {
			inputContent.push_back(inputStr);
		}
		fp.close();
		string type = "Norm_ROI_";
		double count_time = 0;
//#pragma omp parallel for num_threads(7)
//#pragma omp parallel for
		for (int i = 0; i < original.size(); i++)//original.size()
		{
			if (original[i].find(".json", 0) != -1)
				continue;

			//	size_t sdf = original[i].find("data_val_" + user_num + "\\", 0);
			size_t sdf = original[i].find("\\original\\");
			string name = original[i].substr(sdf + 4 + 6 + 4, 13);

			//	sdf = original[i].find("\\Frame");
			//	name = original[i].substr(sdf+1 );
			Mat che = imread(store_path + type + name + ".png", 0);
			if (che.empty() == 0)
			{
				continue;
				waitKey(10);
			}
			bool check_name = 0;
			for (int z = 0; z < inputContent.size(); z++)
			{
				if (name == inputContent[z])
				{
					check_name = 1;
				}
			}
			if (check_name == true)
			{
				continue;
			}
			//if (name == "012_l_940_028")
			//	continue;
			//if (name == "012_l_940_049")
			//	continue;
			//if (name == "012_l_940_051")
			//	continue;
			//size_t sdffff = name.find("012_l_940");
			//if (sdffff>=0)
			//	continue;

			clock_t start, end;
			start = clock();
			Mat g_utimage_in = imread(original[i], 0);
			Mat g_utimage_mask = imread(mask_path+"\\predict_"+ original[i].substr(sdf + 4 + 6 + 4), 0);
			
			cout << original[i] << endl;
			//cout << mask[i] << endl;
			if (g_utimage_in.rows > g_utimage_in.cols)
			{
				g_utimage_in = transpose(g_utimage_in);
			}
			Mat g_utimage_OTSU = g_utimage_mask.clone();
			/*	double thresh = 0.85;
				Mat g_utimage_gaussian;
				GaussianBlur(g_utimage_in, g_utimage_gaussian, Size(5, 5), 0, 0);
				threshold(g_utimage_gaussian, g_utimage_OTSU, thresh, 255, CV_THRESH_OTSU);
				for (int i = 0; i < g_utimage_OTSU.rows; i++)
				{
					for (int j = 0; j < g_utimage_OTSU.cols; j++)
					{
						if (g_utimage_OTSU.at<uchar>(i, j) == 255)
						{
							g_utimage_OTSU.at<uchar>(i, j) = 0;
						}
						else if (g_utimage_OTSU.at<uchar>(i, j) == 0)
						{
							g_utimage_OTSU.at<uchar>(i, j) = 255;
						}
					}
				}
				//	Mat g_utimage_mask;
				Mat kernal(5, 5, CV_8U);
				dilate(g_utimage_OTSU, g_utimage_OTSU, kernal);
				dilate(g_utimage_OTSU, g_utimage_OTSU, kernal);
				erode(g_utimage_OTSU, g_utimage_OTSU, kernal);
				erode(g_utimage_OTSU, g_utimage_OTSU, kernal);
				labeling(g_utimage_OTSU, g_utimage_mask);
				g_utimage_mask = labeling_replace(g_utimage_mask);
				medianBlur(g_utimage_mask, g_utimage_mask, 9);
				imwrite("C:\\Deep_learning\\dataset\\Palm-image\\data_dataset_voc\\SegmentationClassPNG\\mask_" + name + ".png", g_utimage_mask);
				imwrite("C:\\Deep_learning\\dataset\\Palm-image\\data_dataset_voc\\JPEGImages\\image_" + name + ".png", g_utimage_in);*/
				/*	if (g_utimage_mask.channels() == 1) {
						Mat_<uchar>::iterator it = g_utimage_mask.begin<uchar>();
						Mat_<uchar>::iterator itend = g_utimage_mask.end<uchar>();
						for (; it != itend; it++) {
							if ((*it)>=200)
							{
								(*it) = 0;
							}
							else
							{
								(*it) = 255;
							}

						}
					}
					if (g_utimage_in.rows > g_utimage_in.cols)
						rotate(g_utimage_in, g_utimage_in, 2);
					if (g_utimage_mask.size() != Size(768, 576))
						resize(g_utimage_mask, g_utimage_mask, Size(768, 576));
					if (g_utimage_in.size() != g_utimage_mask.size())
						if (g_utimage_in.size() != Size(768, 576))
							resize(g_utimage_in, g_utimage_in, Size(768, 576));
							threshold(g_utimage_mask, g_utimage_mask, 2, 255, CV_THRESH_BINARY);*/




			size_t hand = original[i].find("_r_", 0);
			string handcheck;
			if (hand != -1)
			{
				handcheck = original[i].substr(hand + 1, 1);
			}
			else
			{
				hand = original[i].find("_l_", 0);
				handcheck = original[i].substr(hand + 1, 1);
			}

			try
			{
				Mat g_utimage_store;
				Mat g_utimage_ROI = catch_palm_ROI(g_utimage_mask, g_utimage_in, g_utimage_store, handcheck, name);
				end = clock();
				count_time = count_time + end - start;
				if (g_utimage_ROI.empty() == false)
				{
					imwrite(store_path + type + name + ".png", g_utimage_ROI);
					hconcat(g_utimage_mask, g_utimage_in, g_utimage_mask);
					cvtColor(g_utimage_mask, g_utimage_mask, CV_GRAY2BGR);
					vconcat(g_utimage_store, g_utimage_mask, g_utimage_mask);
					imwrite(log_path + type + "_log_" + name + ".png", g_utimage_mask);
				}
			}

			catch (const char* str)
			{
				cout << "錯誤:" << str << endl;
				fstream  fp;  //宣 告一個fstream物 件
				fp.open("error_log.txt", ios::out);
				if (!fp) //如果開啟檔案失敗，fp為0；成功，fp為非0     
				{
					cout << "Fail to open error_log " << endl;
				}
			}

		}
		inputContent.clear();
		vector <string>().swap(inputContent);
	}
	
	
	return 0;
}


int ncon(int p[9])
{
	int	i, i1, i2;
	int q[9];
	int n = 0;

	for (i = 0; i < 9; i++) {
		if ((p[i] == 1) || (p[i] == -1)) q[i] = 0;
		else q[i] = 1;
	}
	for (i = 1; i < 9; i += 2) {
		i1 = i + 1;
		i2 = i + 2;
		if (i2 == 9) i2 = 1;
		n = n + q[i] - q[i] * q[i1] * q[i2];
	}
	return n;
}

Mat thinning(Mat a_timage_in)
{
	Mat a_timage_out = a_timage_in.clone();//(a_timage_in.rows, a_timage_in.cols, CV_8U);
	int flg = 1;
	int i, j, k, n;
	int p[9];	/* 圖形:1背景:0，背景候補:-1 */

				/*for (i = 0; i < a_timage_in.rows; i++)
				for (j = 0; j < a_timage_in.cols; j++)
				a_timage_out.at<uchar>(i, j) = a_timage_in.at<uchar>(i, j);*/
	while (flg != 0) {
		flg = 0;
		for (i = 1; i <a_timage_in.rows - 1; i++) {
			for (j = 1; j < a_timage_in.cols - 1; j++) {
				p[0] = a_timage_out.at<uchar>(i, j);
				p[1] = a_timage_out.at<uchar>(i, j + 1);
				p[2] = a_timage_out.at<uchar>(i - 1, j + 1);
				p[3] = a_timage_out.at<uchar>(i - 1, j);
				p[4] = a_timage_out.at<uchar>(i - 1, j - 1);
				p[5] = a_timage_out.at<uchar>(i, j - 1);
				p[6] = a_timage_out.at<uchar>(i + 1, j - 1);
				p[7] = a_timage_out.at<uchar>(i + 1, j);
				p[8] = a_timage_out.at<uchar>(i + 1, j + 1);
				for (k = 0; k < 9; k++) {
					if (p[k] == 255) p[k] = 1;
					else if (p[k] == 0) p[k] = 0;
					else                   p[k] = -1;
				}
				/* 條件1:圖形的一部分 */
				if (p[0] != 1) continue;
				/*條件2:境界像素(4個鄰近像素有1個以上是背景)*/
				if (p[1] * p[3] * p[5] * p[7] != 0) continue;
				/*條件3:保留端點(8個鄰近像素有2個以上是圖形)*/
				n = 0;
				for (k = 1; k < 9; k++) if (p[k] != 0) n++;
				if (n < 2) continue;
				/*條件4:保留獨立點(8個鄰近像素有1個以上圖形)*/
				n = 0;
				for (k = 1; k < 9; k++) if (p[k] == 1) n++;
				if (n < 1) continue;
				/*條件5:保留連結性(8個連結數為1)*/
				if (ncon(p) != 1) continue;
				/*條件6:線條寬度為2時，只去除單邊(8個鄰近像素是否並非全部是-1，如果是-1時，其值為0，8個連
				結數為1)*/
				n = 0;
				for (k = 1; k < 9; k++) {
					if (p[k] != -1) n++;
					else if (p[k] == -1) {
						p[k] = 0;
						if (ncon(p) == 1) n++;
						p[k] = -1;
					}
				}
				if (n < 8) continue;
				/*條件1-6全部滿足時的削除對象 */
				a_timage_out.at<uchar>(i, j) = 128;
				flg++;
			}
		}
		for (i = 1; i <a_timage_in.rows - 1; i++)
			for (j = 1; j < a_timage_in.cols - 1; j++)
				if (a_timage_out.at<uchar>(i, j) == 128) a_timage_out.at<uchar>(i, j) = 0;
	}
	return a_timage_out;
}


int Track_local_point(Point &P_in, Point &P_old, Mat a_timage_in)//決定下一個要計算歐式距離的點
{
	Mat a_image_ROI = a_timage_in(Rect(P_in.x - 1, P_in.y - 1, 3, 3));
	vector<int> P_x;
	vector<int> P_y;
	vector<int> P_Judge_x;//可能的下一點陣列x
	vector<int> P_Judge_y;//可能的下一點陣列y
	Point P_corner[4];
	int a = 0;
	for (int i = -1;i < 2; i += 2)//抓出3*3的四個角座標
	{
		for (int j = -1; j < 2; j += 2)
		{
			P_corner[a].x = P_in.x + i;
			P_corner[a].y = P_in.y + j;
			a++;
		}
	}
	Point P_new(0, 0);
	for (int i = 0; i < a_image_ROI.rows; i++)//統計總共有多少個可能的下一點
	{
		for (int j = 0; j < a_image_ROI.cols; j++)
		{
			if (a_image_ROI.at<uchar>(i, j) == 255)
			{
				P_x.push_back(j-1+P_in.x);
				P_y.push_back(i-1+P_in.y);
			}
		}
	}

	for (int x = 0; x < P_x.size(); x++)//把中心點和前一點剔除可能得下一點
	{
		if (P_x[x] != P_in.x || P_y[x] != P_in.y)
			if (P_x[x] != P_old.x || P_y[x] != P_old.y)
			{
				P_Judge_x.push_back(P_x[x]);
				P_Judge_y.push_back(P_y[x]);
			}
	}
	
	if (P_Judge_x.size() > 1)
	{	
		int watch = 0;
		for (int i = 0; i < P_Judge_x.size(); i++)//剔除在前一個追蹤點旁邊的不可能點
		{
			for (int j = 0; j < 4; j++)
			{
				if ((abs(P_Judge_x[i] - P_old.x) + abs(P_Judge_y[i] - P_old.y)) == 1)
				{
					//P_new.x = P_Judge_x[i];
					//P_new.y = P_Judge_y[i];
					P_Judge_x.erase(P_Judge_x.begin() + i);
					P_Judge_y.erase(P_Judge_y.begin() + i);
					watch++;
				}
				if (P_Judge_x.size() < 2)
					break;
			}	
		}
		if (watch > 0)
		{
			P_new.x = P_Judge_x[0];
			P_new.y = P_Judge_y[0];
		}
		else if (P_Judge_x.size() != 0)
		{
			for (int i = 0; i < P_Judge_x.size(); i++)//從離追蹤點最遠的角點選擇下一點
			{
				for (int j = 0; j < 4; j++)
				{
					if (P_Judge_x[i] == P_corner[j].x&&P_Judge_y[i]==P_corner[j].y)
					{
						P_new.x = P_Judge_x[i];
						P_new.y = P_Judge_y[i];
						break;
					}
				}
			}
		}
		
	}
	else if(P_Judge_x.size()!=0)//只有一個可能點
	{
		P_new.x = P_Judge_x[0];
		P_new.y = P_Judge_y[0];
	}
	P_Judge_x.clear();
	P_Judge_y.clear();
	P_x.clear();
	P_y.clear();
	vector<int>().swap(P_x);
	vector<int>().swap(P_y);
	vector<int>().swap(P_Judge_x);
	vector<int>().swap(P_Judge_y);
	if (P_new.x == 0 && P_new.y == 0)//完全沒有下一點
		return-1;
	else
	{
		P_old = P_in;
		P_in = P_new;
		return 1;
	} 
}
double Euclidean_distance(Point P_ref, Point detection)//計算歐式距離，P_ref為原點
{
	double d = 0;
	d = sqrt(pow((P_ref.x - detection.x), 2) + pow((P_ref.y - detection.y), 2));
	return d;
}
int wave_peak_valley(Mat a_timage_in, Point &check_point, int check_length)//檢查這個點是否為波峰或波谷，波峰回傳1，波谷回傳2，都不是回傳0
{
	double center = a_timage_in.at<float>(0, check_point.x);
	int counter_peak = 0;
	int counter_valley = 0;
	int reg = 0;

	for (int i = check_point.x + check_length / 2 * -1; i < check_point.x + check_length / 2; i++)
	{
		reg = i;
		if (i < 0)	continue;
		if (a_timage_in.at<float>(0, i) <= center)
			//		if (a_timage_in.at<float>(0, i + check_point.x) <= center)
		{
			counter_peak++;
		}
		if (a_timage_in.at<float>(0, i) >= center)
		{
			counter_valley++;
		}
		i = reg;
	}
	if (counter_peak == check_length)
	{
		return 1;
	}
	else if (counter_valley == check_length)
	{
		return 2;
	}
	else
	{
		return 0;
	}
}

void find_start_track_point_main(Mat &a_timage_canny, Point &P_ref, Point &detection,int head,int left)//找尋最上面最右邊的點當作初始找尋點
{
	int watchdog = 0;
	int count = 2;
	while (watchdog == 0)
	{
		for (int i = 0; i < a_timage_canny.cols / 3+50; i++)
		{
			int j = P_ref.x - count;
			if (a_timage_canny.at<uchar>(i, j) == 255)
			{
				detection.x = j;
				detection.y = i;
				watchdog++;
				break;
			}
		}
		count++;
	}
	
	/*int max_j = 0;
	for (int i = head; i < head + 80; i++)
	{
		for (int j = left; j < P_ref.x; j++)
		{
			if (a_timage_canny.at<uchar>(i, j) == 255)
			{
				if (j > max_j)
				{
					max_j = j;
					detection.x = j;
					detection.y = i;
					//break;
				}
			}
		}
	}*/
	for (int i = detection.y + 1; i < detection.y + 5; i++)//將找到點上的下方清空避免重複
	{
		a_timage_canny.at<uchar>(i, detection.x) = 0;
	}
}

void track_plamedge2image_main(int &track_check, int &watchdog, Point &detection, Point &detection_old, Mat &a_timage_canny, Mat &euc_distance, Mat &reg, Point &P_ref, int &euc_counter)
//計算整個手掌的歐式距離，並存入圖片陣列中
{
	while (track_check == 1 && watchdog == 0)
	{
		track_check = Track_local_point(detection, detection_old, a_timage_canny);
		if (track_check == 1)
		{
			euc_distance.at<float>(0, euc_counter) = Euclidean_distance(P_ref, detection);
			euc_distance.at<float>(1, euc_counter) = detection.x;
			euc_distance.at<float>(2, euc_counter) = detection.y;
			euc_counter++;
			reg.at<uchar>(detection) = 255;
		}
		if (euc_counter == 3038)
		{
			int fdadadde = 0;
			fdadadde++;
		}
			
	}
}

void near_point_find_main(Mat a_timage_canny, int detection_near_num, Point &detection, Point &detection_old, int &watchdog, int &track_check, Mat &euc_distance, int &euc_counter, Mat &reg, Point P_ref)
//找尋中斷點最近的那個點，使用中斷點以下的點尋找，並計算出各點的歐式距離，最近的點即判斷為為下一點。並且包含了後續歐式距離的找尋任務
{
	Mat near_euc_length(3, 2500, CV_32F);
	int near_x = 0;
	int nead_y = 0;
	int near_counter = 0;
	float near_euc_num = 500000.0;
	int near_euc_position = 0;
	int a;//代替i
	Point ori_detection = detection;
	for (int i = detection.y + 1; i < detection.y + detection_near_num; i++)
	{
		for (int j = detection.x - detection_near_num / 2; j < detection.x + detection_near_num / 2; j++)
		{
			a = i;
			if (i >= a_timage_canny.rows)
				a = a_timage_canny.rows - 1;
			detection_old.x = j;
			detection_old.y = a;
			if (a_timage_canny.at<uchar>(detection_old) == 255 && detection != detection_old)
			{
				near_euc_length.at<float>(0, near_counter) = Euclidean_distance(detection, detection_old);
				near_euc_length.at<float>(1, near_counter) = detection_old.x;
				near_euc_length.at<float>(2, near_counter) = detection_old.y;
				if (near_euc_length.at<float>(0, near_counter) <= near_euc_num)
				{
					near_euc_num = near_euc_length.at<float>(0, near_counter);
					near_euc_position = near_counter;
				}
				near_counter++;
			}
		}
	}
	detection.x = near_euc_length.at<float>(1, near_euc_position);
	detection.y = near_euc_length.at<float>(2, near_euc_position);
	if (detection.x > 0 && detection.y > 0)
	{
		for (int i = detection.y - 2; i < detection.y + 2; i++)
		{
			for (int j = detection.x + 1; j < detection.x + 4; j++)
			{
				a_timage_canny.at<uchar>(i, j) = 0;
			}
		}
		watchdog = 0;
		track_check = 1;
		detection_old = detection;
		track_plamedge2image_main(track_check, watchdog, detection, detection_old, a_timage_canny, euc_distance, reg, P_ref, euc_counter);
	}
}

void find_peak_vally_point_main(int &check_length, int &euc_counter, Point &P_reg, Mat &euc_distance, float *peak_x, float *peak_y, int &peak_counter, float *valley_x, float *valley_y, int &valley_counter)
//找出歐式距離中，手掌歐式距離的最高和最低點(RDF圖像)
{
	int not_find_near_distance = 30;
	for (int z = check_length / 2; z < euc_counter - check_length / 2; z++)
	{
		P_reg.x = z;
		int receive = wave_peak_valley(euc_distance, P_reg, check_length);
		if (receive == 1)
		{
			peak_x[peak_counter] = euc_distance.at<float>(1, z);
			peak_y[peak_counter] = euc_distance.at<float>(2, z);
			peak_counter++;
			if (abs(peak_x[peak_counter - 1] - peak_x[peak_counter - 2]) < not_find_near_distance && abs(peak_y[peak_counter - 1] - peak_y[peak_counter - 2]) < not_find_near_distance)//避免抓到的最低點為相鄰點
			{
				peak_x[peak_counter] = 0;
				peak_y[peak_counter] = 0;
				peak_counter--;
			}
		}
		if (receive == 2)
		{
			valley_x[valley_counter] = euc_distance.at<float>(1, z);
			valley_y[valley_counter] = euc_distance.at<float>(2, z);
			valley_counter++;
			if (abs(valley_x[valley_counter - 1] - valley_x[valley_counter - 2]) < not_find_near_distance && abs(valley_y[valley_counter - 1] - valley_y[valley_counter - 2]) < not_find_near_distance)//避免抓到的最低點為相鄰點
			{
				valley_x[valley_counter] = 0;
				valley_y[valley_counter] = 0;
				valley_counter--;
			}
		}
	}
}
#include <fstream>

void writeCSV(string filename, Mat m)
{
	ofstream myfile;
	myfile.open(filename.c_str());
	myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
	myfile.close();
}
static int aoiGravityCenter(Mat in, Point &center)//找出圖片重心
{
	//if(!src)IplImage *src
	// return GRAVITYCENTER__SRC_IS_NULL;
	//IplImage *src= IplImage(in);
	//IplImage *src = &IplImage(in);
	IplImage* img = new IplImage(in);
	double m00, m10, m01;
	CvMoments moment;
	cvMoments(img, &moment, 1);
	m00 = cvGetSpatialMoment(&moment, 0, 0);
	if (m00 == 0)
		return 1;
	m10 = cvGetSpatialMoment(&moment, 1, 0);
	m01 = cvGetSpatialMoment(&moment, 0, 1);
	center.x = (int)(m10 / m00);
	center.y = (int)(m01 / m00);
	return 0;
}


Mat catch_palm_ROI(Mat &a_timage_in, Mat a_timage_plam,Mat &strong_image,string hand,string name)/*輸入乾淨的二質化圖片，並回傳ROI圖片，in=二質化圖片,plam=手掌原圖,hand_check=輸入現在是左手還是右手圖片,strong_img可將圖片回傳主程式的暫存器*/
{
	Mat kernal(5, 5, CV_8U);
	Mat kernal_2(3, 3, CV_8U);
	erode(a_timage_in, a_timage_in, kernal);
	erode(a_timage_in, a_timage_in, kernal);
	erode(a_timage_in, a_timage_in, kernal);
	erode(a_timage_in, a_timage_in, kernal);
	erode(a_timage_in, a_timage_in, kernal_2);
	dilate(a_timage_in, a_timage_in, kernal_2);
	dilate(a_timage_in, a_timage_in, kernal);
	dilate(a_timage_in, a_timage_in, kernal);
	dilate(a_timage_in, a_timage_in, kernal);
	dilate(a_timage_in, a_timage_in, kernal);
	

	float head = 1000000, leg = 0, right = 0, left = 1000000;//使用這四個參數抓出二質化圖片的一個正方形大小
	for (int i = 0; i < a_timage_in.rows; i++)
	{
		for (int j = 0; j < a_timage_in.cols; j++)
		{
			if (a_timage_in.at<uchar>(i, j) == 255)
			{
				if (i < head)
				{
					head = i;
				}
				if (i > leg)
				{
					leg = i;
				}
				if (j<left)
				{
					left = j;
				}
				if (j > right)
				{
					right = j;
				}
			}
		}
	}
	/*//a_timage_out已經是二質化的整隻手掌的圖片
	Point center(a_timage_in.cols / 2, a_timage_in.rows / 2);
	//Point center((right - left) * 4 / 5 + left, (leg - head) / 2 + head);
	Point edge(center.x, center.y);
	for (int i = 0; i < a_timage_in.rows; i++)
	{
		for (int j = center.x; j < a_timage_in.cols; j++)//找二質化圖片中心點的for迴圈
		{
			if (a_timage_in.at<uchar>(i, j) == 0)
			{
				if (j > edge.x)
				{
					edge.x = j;
				}
			}
		}
	}
	Point middle((edge.x - center.x) / 2 + center.x, center.y);
	Point middle;
	aoiGravityCenter(a_timage_in, middle);
	Point center(a_timage_in.cols / 2, a_timage_in.rows / 2);*/
//	cout << abs(left - right) << endl;
	/*if (abs(left-right)>550)
	{
		for (int i = 0; i < a_timage_in.rows; i++)
		{
			for (int j = left + 550; j < a_timage_in.cols; j++)
			{
				a_timage_in.at<uchar>(i, j) = 0;
			}
		}
	}
	right = left + 550;*/
	Point center;
	aoiGravityCenter(a_timage_in, center);
	Point middle(abs(left - right) * 9 / 10 + left,abs(leg-head)/2+head);
	Mat a_timage_noWrist = a_timage_in.clone();
	for (int i = 0; i < a_timage_in.rows; i++)//將手腕部分去除的for迴圈
	{
		for (int j = 0; j < a_timage_in.cols; j++)
		{
			if (j>middle.x)
			{
				a_timage_noWrist.at<uchar>(i, j) = 0;
			}
		}
	}
//	imwrite("C:\\Users\\raychen\\Desktop\\ROI\\noWris" + name + ".png", a_timage_noWrist);
	Point P = center;
	Point P_ref = middle;
	Mat a_timage_canny = a_timage_in.clone();
	//	Canny(a_timage_noWrist, a_timage_canny, 50, 50, 3);
	//	a_timage_canny = thinning(a_timage_canny);
	Mat a_canny_test;
	Mat a_canny_test_2;
	medianBlur(a_timage_noWrist, a_canny_test, 5);
	Canny(a_canny_test, a_canny_test_2, 50, 50, 3);
	a_canny_test_2 = thinning(a_canny_test_2);
	a_timage_canny = a_canny_test_2.clone();
	int patch_start = 0, patch_end = 0;
	//	imwrite("binary hand.bmp", a_canny_test);
	
	for (int i = 0; i < a_timage_canny.rows; i += a_timage_canny.rows - 1)
	{
		for (int j = 0; j < a_timage_canny.cols; j++)//為了手掌超出圖片邊緣時，為消失的區域補點，直接將點補上去，並無特別方法，並位移到下方一個pixel處
		{
			if (a_timage_canny.at<uchar>(i, j) == 255 && a_timage_canny.at<uchar>(i, j - 1) == 0 && a_timage_canny.at<uchar>(i, j + 1) == 0)
			{
				patch_start = j;
				j++;
				while (a_timage_canny.at<uchar>(i, j) == 0)
				{
					j++;
					if (a_timage_canny.at<uchar>(i, j) == 255)
					{
						patch_end = j + 1;
						for (int a = patch_start; a < patch_end; a++)
						{
							if (i == 0)
							{
								a_timage_canny.at<uchar>(i + 1, a) = 255;
								a_timage_canny.at<uchar>(i, a) = 0;
							}
							else if (i == a_timage_canny.rows - 1)
							{
								a_timage_canny.at<uchar>(i - 1, a) = 255;
								a_timage_canny.at<uchar>(i, a) = 0;
							}
						}
						break;
					}
					if (j > a_timage_canny.cols - 2)
					{
						break;
					}
				}
			}
		}
	}
	for (int j = 0; j < a_timage_canny.cols; j += a_timage_canny.cols - 1)
	{
		for (int i = 0; i < a_timage_canny.rows; i++)//為了手掌超出圖片邊緣時，為消失的區域補點，直接將點補上去，並無特別方法，並位移到下方一個pixel處
		{
			if (a_timage_canny.at<uchar>(i, j) == 255 && a_timage_canny.at<uchar>(i - 1, j) == 0 && a_timage_canny.at<uchar>(i + 1, j) == 0)
			{
				patch_start = i;
				i++;
				while (a_timage_canny.at<uchar>(i, j) == 0)
				{
					i++;
					if (a_timage_canny.at<uchar>(i, j) == 255)
					{
						patch_end = i + 1;
						for (int a = patch_start; a < patch_end; a++)
						{
							if (j == 0)
							{
								a_timage_canny.at<uchar>(a, j + 1) = 255;
								a_timage_canny.at<uchar>(a, j) = 0;
							}
							else if (i == a_timage_canny.cols - 1)
							{
								a_timage_canny.at<uchar>(a, j - 1) = 255;
								a_timage_canny.at<uchar>(a, j) = 0;
							}
						}
						break;
					}
					if (i > a_timage_canny.rows - 2)
					{
						break;
					}
				}
			}
		}
	}
	for (int j = 0; j < a_timage_canny.cols; j++)
	{
		if (a_timage_canny.at<uchar>(0, j) == 255)
		{
			a_timage_canny.at<uchar>(0, j) = 0;
			a_timage_canny.at<uchar>(1, j) = 255;
		}
		if (a_timage_canny.at<uchar>(a_timage_canny.rows - 1, j) == 255)
		{
			a_timage_canny.at<uchar>(a_timage_canny.rows - 1, j) = 0;
			a_timage_canny.at<uchar>(a_timage_canny.rows - 1 - 1, j) = 255;;
		}
	}
	for (int i = 0; i < a_timage_canny.rows; i++)
	{
		if (a_timage_canny.at<uchar>(i, 0) == 255)
		{
			a_timage_canny.at<uchar>(i, 0) = 0;
			a_timage_canny.at<uchar>(i, 1) = 255;
		}
		if (a_timage_canny.at<uchar>(i, a_canny_test.cols - 1) == 255)
		{
			a_timage_canny.at<uchar>(i, a_canny_test.cols - 1) = 0;
			a_timage_canny.at<uchar>(i, a_canny_test.cols - 2) = 255;
		}
	}




	for (int i = 0; i < a_timage_canny.rows; i++)//將P_ref向左2點右邊的canny圖像刪除
	{
		for (int j = 0; j < a_timage_canny.cols; j++)
		{
			if (j > P_ref.x - 2)
			{
				a_timage_canny.at<uchar>(i, j) = 0;
			}
		}
	}
	Point detection(P_ref.x, P_ref.y);

	find_start_track_point_main(a_timage_canny, P_ref, detection, head, left);
	for (int i = 0; i < a_timage_canny.rows; i++)//將起始以點右邊的canny圖像刪除
	{
		for (int j = 0; j < a_timage_canny.cols; j++)
		{
			if (j > detection.x)
			{
				a_timage_canny.at<uchar>(i, j) = 0;
			}
		}
	}
	Mat reg(a_timage_canny.rows, a_timage_canny.cols, a_timage_canny.type(), Scalar(0));
	Mat euc_distance(3, a_timage_canny.rows*a_timage_canny.cols / 10, CV_32F, Scalar(0));//用來儲存歐式距離的資料，第一通道存歐式距離，第二通道存x，第三通道存y
	int euc_counter = 0;
	int watchdog = 0;
	int track_check = 1;
	Point detection_old = detection;
	a_timage_canny.at<uchar>(detection) = 128;
	try 
	{
		track_plamedge2image_main(track_check, watchdog, detection, detection_old, a_timage_canny, euc_distance, reg, P_ref, euc_counter);
	}
	catch (const char* str)
	{
		cout << "錯誤:" << str << endl;
		fstream  fp;  //宣 告一個fstream物 件
		fp.open("error_log.txt", ios::out);
		if (!fp) //如果開啟檔案失敗，fp為0；成功，fp為非0     
		{
			cout << "Fail to open error_log " << endl;
		}
		fp << name << endl;
		fp.close();
	}
	int detection_near_num = 180*2.96296296296;//檢測附近幾個pxiel會出現下一點
	/*while (euc_counter<1200)
	{
		near_point_find_main(a_timage_canny, detection_near_num, detection, detection_old, watchdog, track_check, euc_distance, euc_counter, reg, P_ref);
	}*/
	Point P_reg(1, 1);
	float peak_x[100];
	float peak_y[100];
	float valley_x[10];
	float valley_y[10];
	int peak_counter = 0;
	int valley_counter = 0;
	int check_length = 100*2.96296296296;//檢查前後總共幾個pixel;


	find_peak_vally_point_main(check_length, euc_counter, P_reg, euc_distance, peak_x, peak_y, peak_counter, valley_x, valley_y, valley_counter);
	Point P1, P2;
	if (valley_counter <= 3)
	{
		near_point_find_main(a_timage_canny, detection_near_num, detection, detection_old, watchdog, track_check, euc_distance, euc_counter, reg, P_ref);
		find_peak_vally_point_main(check_length, euc_counter, P_reg, euc_distance, peak_x, peak_y, peak_counter, valley_x, valley_y, valley_counter);
	}
	/*if (valley_counter <= 3)
	{
		near_point_find_main(a_timage_canny, detection_near_num, detection, detection_old, watchdog, track_check, euc_distance, euc_counter, reg, P_ref);
		find_peak_vally_point_main(check_length, euc_counter, P_reg, euc_distance, peak_x, peak_y, peak_counter, valley_x, valley_y, valley_counter);
	}
	if (valley_counter <= 3)
	{
		near_point_find_main(a_timage_canny, detection_near_num, detection, detection_old, watchdog, track_check, euc_distance, euc_counter, reg, P_ref);
		find_peak_vally_point_main(check_length, euc_counter, P_reg, euc_distance, peak_x, peak_y, peak_counter, valley_x, valley_y, valley_counter);
	}*/
	

//	writeCSV("C:\\Users\\raychen\\Desktop\\ROI\\Euilcdis" + name + ".csv", euc_distance);
	for (int i = 0; i < 10; i++)
	{
		for (int j = i + 1; j < 10; j++)
		{
			if (valley_x[i] == valley_x[j] && valley_y[i] == valley_y[j])
			{
				valley_x[j] = NULL;
				valley_y[j] = NULL;
			}
			if (peak_x[i] == peak_x[j] && peak_y[i] == peak_y[j])
			{
				peak_x[j] = NULL;
				peak_y[j] = NULL;
			}
			/*if (detection.x - 10 <= valley_x[i])
			{
				if (valley_x[i] <= detection.x + 10)
				{
					valley_x[j] = NULL;
					valley_y[j] = NULL;
				}
			}
			if (detection.y - 10 <= valley_y[i] <= detection.y + 10)
			{
				if (valley_y[i] <= detection.y + 10)
				{
					valley_x[j] = NULL;
					valley_y[j] = NULL;
				}
			}*/
		}
	}
	for (int i = 0; i < 10; i++)
	{
		for (int j = i + 1; j < 10; j++)
		{
			if (valley_x[j] == NULL&&valley_y[j] == NULL)
			{
				if (j + 1 < 10)
				{
					valley_x[j] = valley_x[j + 1];
					valley_y[j] = valley_y[j + 1];
					valley_x[j + 1] = NULL;
					valley_y[j + 1] = NULL;
				}			
			}
		}
	}
	

	string hand_check;
	vector <int> axis_x;
	vector <int> axis_y;
	//for (int i = 0; i < 9; i++)
	//{
	//	valley_x[i] = valley_x[i + 1];
	//	valley_y[i] = valley_y[i + 1];
	//}
	for (int i = 0; i < 100; i++)
	{
		if (valley_x[i] > 0)
			axis_x.push_back(valley_x[i]);
		else
		{
			break;
		}
	}
	for (int i = 0; i < 100; i++)
	{
		if (valley_y[i] > 0)
			axis_y.push_back(valley_y[i]);
		else
		{
			break;
		}
	}
	if (axis_x.size() < 4)
	{
		//waitKey(0);
		axis_x.clear();
		vector <int>().swap(axis_x);
		axis_y.clear();
		vector <int>().swap(axis_y);
		throw "未抓到4個谷點";
		return Mat();
	}
	else if (axis_y.size() < 4)
	{
		//waitKey(0);
		axis_x.clear();
		vector <int>().swap(axis_x);
		axis_y.clear();
		vector <int>().swap(axis_y);
		throw "未抓到4個谷點";
		return Mat();
	}
	else
	{
		Mat reg_train(1, 8, CV_32F, Scalar(0));
	/*	for (int i = 0; i < 8; i += 2)
		{
			reg_train.at<float>(0, i) = axis_x.at(i / 2);
			reg_train.at<float>(0, i + 1) = axis_y.at(i / 2);
		}
		Ptr<SVM> svm = SVM::load("hand_check.sav");
		int check = svm->predict(reg_train);
		axis_x.clear();
		vector <int>().swap(axis_x);
		axis_y.clear();
		vector <int>().swap(axis_y);
		if (check == 0)
		{
			hand_check = "r";
		}
		else if (check == 1)
			hand_check = "l";
		if (hand_check != hand)
			hand_check = hand;
			waitKey(10);*/
	//	hand_check = "r";
		//string hand_check = hand;

		/******************************************下面先假定不會有只抓到三個點的情況出現*************************************/
		hand_check = hand;
		if (hand_check == "l")
		{
			P1.x = valley_x[2];
			P1.y = valley_y[2];
			P2.x = valley_x[0];
			P2.y = valley_y[0];
		}
		if (hand_check == "r")
		{
			/*if (valley_x[3] <= 0 && valley_y[3] <= 0)
			{
				P1.x = valley_x[0];
				P1.y = valley_y[0];
				P2.x = valley_x[2];
				P2.y = valley_y[2];
			}
			else*/
			{
				P1.x = valley_x[1];
				P1.y = valley_y[1];
				P2.x = valley_x[3];
				P2.y = valley_y[3];
			}
		}
		Mat a_draw_point(a_canny_test.rows, a_canny_test.cols, CV_8UC3);//= a_canny_test.clone();
		cvtColor(a_canny_test, a_draw_point, CV_GRAY2BGR);
		//	a_draw.convertTo(a_draw, CV_32FC3, 1, 0);
		Point center_1;
		for (int i = 0; i < 5; i++)
		{
			center_1.x = valley_x[i];
			center_1.y = valley_y[i];
			circle(a_draw_point, center_1, 5, Scalar(0, 255, 0), 3, 8, 0);
		}
		for (int i = 0; i < 5; i++)
		{
			center_1.x = peak_x[i];
			center_1.y = peak_y[i];
			circle(a_draw_point, center_1, 5, Scalar(0, 0, 255), 3, 8, 0);
		}
		//	imwrite("peak_vally.bmp", a_draw);
		//	hconcat(g_utimage_label, g_utimage_in, g_utimage_label);
		Mat free = a_timage_plam.clone();
		free.at<uchar>(P2) = 255;
		free.at<uchar>(P1) = 255;
		double xp2 = P2.x, xp1 = P1.x, yp2 = P2.y, yp1 = P1.y;
		Mat a_draw_line(a_canny_test.rows, a_canny_test.cols, CV_8UC3);
		cvtColor(a_timage_plam, a_draw_line, CV_GRAY2BGR);
		circle(a_draw_line, P1, 5, Scalar(0, 0, 255), 3, 8, 0);
		circle(a_draw_line, P2, 5, Scalar(0, 0, 255), 3, 8, 0);
		line(a_draw_line, P1, P2, Scalar(0, 255, 0), 2, 8, 0);
		int length = Euclidean_distance(P1, P2);
		if (hand_check == "r")
		{
			center_1.x = P1.x;
			center_1.y = P1.y + length;
			line(a_draw_line, P1, center_1, Scalar(255, 255, 255), 2, 8, 0);
		}
		else if (hand_check == "l")
		{
			center_1.x = P1.x;
			center_1.y = P1.y - length;
			line(a_draw_line, P1, center_1, Scalar(255, 255, 255), 2, 8, 0);
		}

		//	imwrite("turning angle.bmp", a_draw);

		double theta = (xp2 - xp1) / (yp2 - yp1);
		reg.at<uchar>(P1) = 128;
		reg.at<uchar>(P2) = 128;
		theta = atan(theta)*-180 / PI;
		int scale = 1;
		Mat a_rotate_plam;
		Mat rot_mat = getRotationMatrix2D(P1, theta, scale);
		warpAffine(a_timage_plam, a_rotate_plam, rot_mat, a_timage_plam.size());
		Point P3;
		if (hand_check == "l")
		{
			P3.x = valley_x[1];
			P3.y = valley_y[1];
		}
		if (hand_check == "r")
		{
			P3.x = valley_x[2];
			P3.y = valley_y[2];
		}
		//int f = Euclidean_distance(P1, P2);
		//int g= Euclidean_distance(P1, P3);

		int P_distance = Euclidean_distance(P1, P2);//150 Euclidean_distance(P1, P2) ;// sqrt(f + g) * 11;//Euclidean_distance(P1, P2)+Euclidean_distance(P1, P3);
		int width_length = P_distance;//*sqrt(2);
		int Increase_length = 0;//Euclidean_distance(P1, P3);
		Mat a_timage_out(P_distance, width_length, a_timage_plam.type(), Scalar(0));
		int a = P_distance - 1, b = 0;
		int secret_moving = 20;//移動j

		if (hand_check == "l")
		{
			for (int i = P1.y + Increase_length / 10 * 4; i > P1.y - P_distance + Increase_length / 10 * 4; i--)
			{
				for (int j = P1.x + secret_moving; j < P1.x + width_length + secret_moving; j++)
				{
					a_timage_out.at<uchar>(a, b) = a_rotate_plam.at<uchar>(i, j);
					b++;
					a_rotate_plam.at<uchar>(i, j) = 255;
				}
				a--;
				b = 0;
			}
		}
		else if (hand_check == "r")
		{
			//	a = P_distance-1, b = P_distance-1;
			a = 0, b = 0;
			for (int i = P1.y; i < P1.y + P_distance; i++)
			{
				for (int j = P1.x + secret_moving; j < P1.x + width_length + secret_moving; j++)
				{
					a_timage_out.at<uchar>(a, b) = a_rotate_plam.at<uchar>(i, j);
					b++;
					a_rotate_plam.at<uchar>(i, j) = 255;
				}
				a++;
				b = 0;
			}
		}

		/*int P_distance = Euclidean_distance(P1, P2);// 160 ;// sqrt(f + g) * 11;//Euclidean_distance(P1, P2)+Euclidean_distance(P1, P3);
		int width_length =  P_distance;// *sqrt(2);
		int Increase_length = 0;// width_length/sqrt(2)*0.3;//Euclidean_distance(P1, P3);
		//Increase_length= Increase_length / 10 * 4;
		Mat a_timage_out(width_length + Increase_length, width_length+ Increase_length, a_timage_plam.type(), Scalar(0));
		int a = P1.y + Increase_length / 2, b = 0;
		int secret_moving =20;//移動j


		if (hand_check == "l")////舊的ROI提取
		{ 
			int row_end = P1.y - P_distance - Increase_length/2;
			int col_end = P1.x + width_length + secret_moving;
			if (row_end < 0)
				row_end = 0;
			if (col_end > a_rotate_plam.cols - 1)
				col_end = a_rotate_plam.cols - 1;
			for (int i = P1.y + Increase_length / 2; i > row_end; i--)
			{
				for (int j = P1.x + secret_moving + Increase_length / 2; j < col_end; j++)
				{
					if (i > a_rotate_plam.rows - 1)
						i = a_rotate_plam.rows - 1;
					if (j > a_rotate_plam.cols - 1)
					{
						j = a_rotate_plam.cols - 1;
					}
					a_timage_out.at<uchar>(a, b) = a_rotate_plam.at<uchar>(i, j);
					b++;
					a_rotate_plam.at<uchar>(i, j) = 255;
				}
				a--;
				b = 0;
			}
		}
		else if (hand_check == "r")
		{
			//	a = P_distance-1, b = P_distance-1;
			a = 0, b = 0;
			int row_end = P1.y + P_distance+ Increase_length / 2;
			int col_end = P1.x + width_length + secret_moving;
			if (row_end > a_rotate_plam.rows - 1)
				row_end = a_rotate_plam.rows - 1;
			if (col_end > a_rotate_plam.cols - 1)
				col_end = a_rotate_plam.cols - 1;
			for (int i = P1.y- Increase_length / 2; i < row_end; i++)
			{
				for (int j = P1.x + secret_moving; j < col_end; j++)
				{
				
					a_timage_out.at<uchar>(a, b) = a_rotate_plam.at<uchar>(i, j);
					b++;
					a_rotate_plam.at<uchar>(i, j) = 255;
				}
				a++;
				b = 0;
			}
		}*/
		hconcat(a_draw_line, a_draw_point, strong_image);
	//	imwrite("C:\\Users\\raychen\\Desktop\\ROI\\draw_line" + name + ".png", a_draw_line);
	//	imwrite("C:\\Users\\raychen\\Desktop\\ROI\\draw_point" + name + ".png", a_draw_point);
		
		a_timage_in = a_rotate_plam;
		if (hand_check == "r")
		{
			center_1.x = P1.x;
			center_1.y = P1.y + length;
			line(a_rotate_plam, P1, center_1, Scalar(255, 0, 128), 2, 8, 0);
		}
		if (hand_check == "l")
		{
			center_1.x = P1.x;
			center_1.y = P1.y - length;
			line(a_rotate_plam, P1, center_1, Scalar(255, 0, 128), 2, 8, 0);
		}
		//imwrite("point.png", a_draw_point);
		//imwrite("line.png", a_draw_line);
		//imwrite("sub_white.png", a_rotate_plam);
	//	imwrite("C:\\Users\\raychen\\Desktop\\ROI\\area" + name + ".png", a_rotate_plam);
		return a_timage_out;
	}
}


