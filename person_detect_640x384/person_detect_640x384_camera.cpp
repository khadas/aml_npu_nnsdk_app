#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <iomanip>

#include <pthread.h>
#include <sys/time.h>
#include <sched.h>
#include <sys/resource.h>


#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include "nn_sdk.h"
#include "nn_util.h"
//#include "nn_demo.h"

#define CAMERA_WEIGHT 1920
#define CAMERA_HIGHT 1080


using namespace std;
using namespace cv;

static const char *sdkversion = "v1.6.2";
static void *context = NULL;
img_classify_out_t *cls_out = NULL;
aml_config config;

char *video_device = NULL;
//cv::Mat img;

pthread_mutex_t mutex4q;

void sdk_version(){
	cout << "SDK version:" << sdkversion << endl;
}

void help(){
	cout << "SDK version:" << sdkversion << endl;
	cout << "Useage ./person_detect_640x384_camera < path to person detect nb file>  < path to camera node> " << endl;
}

float Float16ToFloat32(const signed short* src , float* dst ,int lenth)
{
    signed int t1;
    signed int t2;
    signed int t3;
    float out;
    int i;
    for (i = 0 ;i < lenth ;i++)
    {
        t1 = src[i] & 0x7fff;                       // Non-sign bits
        t2 = src[i] & 0x8000;                       // Sign bit
        t3 = src[i] & 0x7c00;                       // Exponent

        t1 <<= 13;                              // Align mantissa on MSB
        t2 <<= 16;                              // Shift sign bit into position

        t1 += 0x38000000;                       // Adjust bias

        t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero

        t1 |= t2;
        *((unsigned int*)&out) = t1;                 // Re-insert sign bit
        dst[i] = out;

    }
    return out;
}


float *dtype_To_F32(nn_output * outdata ,int sz)
{                                                                                                    
    int stride, fl, i, zeropoint;
    float scale;
    unsigned char *buffer_u8 = NULL;
    signed char *buffer_int8 = NULL;
    signed short *buffer_int16 = NULL;
    float *buffer_f32 = NULL;

    buffer_f32 = (float *)malloc(sizeof(float) * sz );

    if (outdata->out[0].param->data_format == NN_BUFFER_FORMAT_UINT8)
    {
        stride = (outdata->out[0].size)/sz;
        scale = outdata->out[0].param->quant_data.affine.scale;
        zeropoint =  outdata->out[0].param->quant_data.affine.zeroPoint;

        buffer_u8 = (unsigned char*)outdata->out[0].buf;
        for (i = 0; i < sz; i++)
        {
            buffer_f32[i] = (float)(buffer_u8[stride * i] - zeropoint) * scale;
        }
    }

    else if (outdata->out[0].param->data_format == NN_BUFFER_FORMAT_INT8)
    {
        buffer_int8 = (signed char*)outdata->out[0].buf;
        if (outdata->out[0].param->quant_data.dfp.fixed_point_pos >= 0)
        {
            fl = 1 << (outdata->out[0].param->quant_data.dfp.fixed_point_pos);
            for (i = 0; i < sz; i++)
            {
                buffer_f32[i] = (float)buffer_int8[i] * (1.0/(float)fl);
            }
        }
        else
        {
            fl = 1 << (-outdata->out[0].param->quant_data.dfp.fixed_point_pos);
            for (i = 0; i < sz; i++)
                buffer_f32[i] = (float)buffer_int8[i] * ((float)fl);
        }
    }

    else if (outdata->out[0].param->data_format == NN_BUFFER_FORMAT_INT16)
    {
        buffer_int16 =  (signed short*)outdata->out[0].buf;
        if (outdata->out[0].param->quant_data.dfp.fixed_point_pos >= 0)
        {
            fl = 1 << (outdata->out[0].param->quant_data.dfp.fixed_point_pos);
            for (i = 0; i < sz; i++)
            {   
                buffer_f32[i] = (float)((buffer_int16[i]) * (1.0/(float)fl));
            }
        }
        else
        {   
			fl = 1 << (-outdata->out[0].param->quant_data.dfp.fixed_point_pos);
            for (i = 0; i < sz; i++)
                buffer_f32[i] = (float)((buffer_int16[i]) * ((float)fl));
        }
    }
    else if (outdata->out[0].param->data_format == NN_BUFFER_FORMAT_FP16 )
    {   
        buffer_int16 = (signed short*)outdata->out[0].buf;
        
        Float16ToFloat32(buffer_int16 ,buffer_f32 ,sz);
    }
    
    else if (outdata->out[0].param->data_format == NN_BUFFER_FORMAT_FP32)
    {   
        memcpy(buffer_f32, outdata->out[0].buf, sz);
    }
    else
    {   
        printf("Error: currently not support type, type = %d\n", outdata->out[0].param->data_format);
    }
    return buffer_f32;
}


int create_network(char *nbfile){

	memset(&config,0,sizeof(aml_config));
	config.path = (const char *)nbfile;
	config.nbgType = NN_NBG_FILE;
	config.modelType = ONNX;
	context = aml_module_create(&config);	
	return 0;
}

void get_input_data_cv(const cv::Mat& sample, uint8_t* input_data, int img_h, int img_w)
{
	int h,w;
	cv::Mat img = sample;
	cv::Mat img2(img_h,img_w,CV_8UC3,cv::Scalar(0,0,0));

	if((float)img.cols/(float)img_w > (float)img.rows/(float)img_h){
		h=1.0*img_w/img.cols*img.rows;
		w=img_w;
		cv::resize(img, img, cv::Size(w,h));
	}else{
		w=1.0*img_h/img.rows*img.cols;
		h=img_h;
		cv::resize(img, img, cv::Size(w,h));
	}

	int top = (img_h - h)/2;
	int bat = (img_h - h + 1)/2;
	int left = (img_w - w)/2;
	int right = (img_w - w + 1)/2;

	cv::copyMakeBorder(img,img2,top,bat,left,right,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));

	uint8_t* img_data = img2.data;
	int hw = img_h * img_w;


	uint32_t reorder[3] = {0,1,2};
	int order = 0;
	int offset=0,tmpdata=0;
	for (int i = 0; i < hw; i++) {
		offset = 3*i;
		for(int j = 0; j < 3; j++){
			order = reorder[j];
			tmpdata = img_data[offset+order];
			input_data[j+offset] = (uint8_t)((tmpdata >  255) ? 255 : (tmpdata < 0) ? 0 : tmpdata);
		}
	}
}

int preprocess_network(const cv::Mat& sample){

	int ret = 0;
	nn_input inData;
//	const char *jpeg_path = NULL;
	uint8_t* rawdata = (uint8_t*)malloc(640*384*3* sizeof(uint8_t));
	get_input_data_cv(sample, rawdata, 384, 640);
	inData.input_index = 0; //this value is index of input,begin from 0
	inData.size = 640*384*3;
	inData.input = rawdata;
	inData.input_type = RGB24_RAW_DATA;
	ret = aml_module_input_set(context,&inData);

	if(rawdata != NULL){
		free(rawdata);
		rawdata = NULL;
		return -1;
	}

	return ret;
}

int postpress_network(const cv::Mat& sample){
	
	aml_output_config_t outconfig;
	person_detect_out_t *pout = NULL;	
	cv::Mat img = sample;

	outconfig.mdType = PERSON_DETECT;
	outconfig.format = AML_OUTDATA_FLOAT32;

	pout =(person_detect_out_t *)aml_module_output_get(context,outconfig);

	cout << "Num:" << pout->detNum << endl;

	if(pout->detNum > 0){
		int i = 0;
		for(i = 0; i < (int)pout->detNum; i++){
			cout << "i:" << i << " x:" << pout->pBox[i].x << " w:" << pout->pBox[i].w << " y:" << pout->pBox[i].y << " h:" << pout->pBox[i].h << " score:" << pout->pBox[i].score <<endl;
			cv::Rect rect(pout->pBox[i].x*img.cols, pout->pBox[i].y*img.rows, pout->pBox[i].w*img.cols, pout->pBox[i].h*img.rows);
			cv::rectangle(img,rect,cv::Scalar(255,0,0),2,2,0);
		}
	}

	return 0;
}

void *thread_camera(void *parameter){
	string str = video_device;
	char *window_name = (char *)"CameraWindow";
	string res=str.substr(10);
	cv::VideoCapture cap(stoi(res));
	cap.set(CV_CAP_PROP_FRAME_WIDTH, CAMERA_WEIGHT);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, CAMERA_HIGHT);
	
	if (!cap.isOpened()) {
		cout << "capture device failed to open!" << endl;
		cap.release();
	}
	
	setpriority(PRIO_PROCESS, pthread_self(), -15);

	cv::namedWindow(window_name);

	while(1){
		cv::Mat frame(CAMERA_HIGHT,CAMERA_WEIGHT,CV_8UC3);
		pthread_mutex_lock(&mutex4q);
		if (!cap.read(frame)) {
			pthread_mutex_unlock(&mutex4q);
			cout<<"Capture read error"<<std::endl;
			break;
		}

		pthread_mutex_unlock(&mutex4q);
		preprocess_network(frame);
		postpress_network(frame);
		cv::imshow(window_name, frame);
		cv::waitKey(1);

		frame.release();
	}

	return 0;
}
int main(int argc,char **argv){

	int i,ret = 0;
	pthread_t tid[2];
	pthread_mutex_init(&mutex4q,NULL);
	if (strcmp(argv[1], "--help") == 0){
		help();
		return 0;
	}
	if(argc < 3){
		help();
		return -1;
	}

	video_device = argv[2];

	sdk_version();

	create_network(argv[1]);

	if (0 != pthread_create(&tid[0], NULL, thread_camera, NULL)) {
		fprintf(stderr, "Couldn't create thread func\n");
		return -1;
	}

	while(1)
	{
		for(i=0; i<(int)sizeof(tid)/(int)sizeof(tid[0]); i++)
		{
			pthread_join(tid[i], NULL);
		}
		sleep(1);
	}


//	preprocess_network(argv[2]);

//	postpress_network();

	ret = aml_module_destroy(context);

	return ret;

}

