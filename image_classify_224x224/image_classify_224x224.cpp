#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <iomanip>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include "nn_sdk.h"
#include "nn_util.h"
//#include "nn_demo.h"

using namespace std;
using namespace cv;

static const char *sdkversion = "v1.6.2";
static void *context = NULL;
static unsigned char *inbuf = NULL;
img_classify_out_t *cls_out = NULL;
aml_config config;



void help(){
	cout << "SDK version:" << sdkversion << endl;
	cout << "Useage ./image_classify_224x224 < path to image_classify nb file>  < path to jpeg file> " << endl;
}

void process_top5_(float *buf,unsigned int num,img_classify_out_t* clsout)
{
    int j = 0;
    unsigned int MaxClass[5]={0};
    float fMaxProb[5]={0.0};

    float *pfMaxProb = fMaxProb;
    unsigned int  *pMaxClass = MaxClass,i = 0;

    for (j = 0; j < 5; j++)
    {
        for (i=0; i<num; i++)
        {
            if ((i == *(pMaxClass+0)) || (i == *(pMaxClass+1)) || (i == *(pMaxClass+2)) ||
                (i == *(pMaxClass+3)) || (i == *(pMaxClass+4)))
            {
                continue;
            }

            if (buf[i] > *(pfMaxProb+j))
            {
                *(pfMaxProb+j) = buf[i];
                *(pMaxClass+j) = i;
            }
        }
    }
    for (i=0; i<5; i++)
    {
        if (clsout == NULL)
        {
            printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);
        }
        else
        {
            clsout->score[i] = fMaxProb[i];
            clsout->topClass[i] = MaxClass[i];
        }
    }
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
	config.modelType = TENSORFLOW;
	inbuf = aml_util_mallocAlignedBuffer(224*224*3);
	config.inOut.inAddr[0] = inbuf;
	config.typeSize = sizeof(aml_config);
	context = aml_module_create(&config);	
	return 0;
}


void get_input_data_cv(char *jpegpath, uint8_t* input_data, int img_h, int img_w)
{
	int h,w;
	cv::Mat img = cv::imread(jpegpath,cv::IMREAD_COLOR);
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

	uint32_t reorder[3] = {2,1,0};
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



int preprocess_network(char *jpegpath){

	int ret = 0;
	nn_input inData;
	uint8_t* rawdata = (uint8_t*)malloc(224*224*3* sizeof(uint8_t));
	get_input_data_cv(jpegpath, rawdata, 224, 224);
	inData.input_index = 0; //this value is index of input,begin from 0
	inData.size = 224*224*3;
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

int postpress_network(){

	int i;
	aml_output_config_t outconfig;
	img_classify_out_t *pout = NULL;
	aml_module_t modelType;

	nn_output *outdata = NULL;

//	outconfig.mdType = IMAGE_CLASSIFY;
//	outconfig.format = AML_OUTDATA_FLOAT32;

//	outdata =(img_classify_out_t *)aml_module_output_get(context,outconfig);

//	cout << "----------Top5----------" << endl;
//	for(i = 0; i < 5; i++){
//		cout << "Top" << i+1 << " class:"<< outdata->topClass[i] << " score:" << pout->score[i] <<endl;
//	}

	outconfig.typeSize = sizeof(aml_output_config_t);
	modelType = CUSTOM_NETWORK;
	outconfig.format = AML_OUTDATA_FLOAT32;
	outdata = (nn_output*)aml_module_output_get(context,outconfig);

	process_top5_((float*)outdata->out[0].buf,outdata->out[0].size/sizeof(float),NULL);

	return 0;
}

int main(int argc,char **argv){

	int ret = 0;
	if (strcmp(argv[1], "--help") == 0){
		help();
		return 0;
	}
	if(argc < 3){
		help();
		return -1;
	}
	
	create_network(argv[1]);
	preprocess_network(argv[2]);
	postpress_network();
	ret = aml_module_destroy(context);
	if(inbuf)aml_util_freeAlignedBuffer(inbuf);

	return ret;

}

