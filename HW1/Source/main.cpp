// Instructions:
// For question 1, only modify function: histogram_equalization
// For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
// For question 3, only modify function: laplacian_pyramid_blending

#include "./header.h"

using namespace std;
using namespace cv;

void help_message(char* argv[])
{
   cout << "Usage: [Question_Number] [Input_Options] [Output_Options]" << endl;
   cout << "[Question Number]" << endl;
   cout << "1 Histogram equalization" << endl;
   cout << "2 Frequency domain filtering" << endl;
   cout << "3 Laplacian pyramid blending" << endl;
   cout << "[Input_Options]" << endl;
   cout << "Path to the input images" << endl;
   cout << "[Output_Options]" << endl;
   cout << "Output directory" << endl;
   cout << "Example usages:" << endl;
   cout << argv[0] << " 1 " << "[path to input image] " << "[output directory]" << endl;
   cout << argv[0] << " 2 " << "[path to input image1] " << "[path to input image2] " << "[output directory]" << endl;
   cout << argv[0] << " 3 " << "[path to input image1] " << "[path to input image2] " << "[output directory]" << endl;
}

// ===================================================
// ======== Question 1: Histogram equalization =======
// ===================================================
void histogram_equalization_single_channel(Mat& channel)
{
   /* Compute PDF*/
   Mat hist;
   float range[] = { 0, 256 };
   const float* histRange = { range };
   bool uniform = true;
   bool accumulate = false;
   int histSize = 256;
   calcHist(&channel, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
   hist.convertTo(hist, CV_32F);
   hist /= hist.at<float>(hist.rows/2);
   /* Compute CDF*/
   Mat cumul_hist(1, 256, CV_32F, Scalar(0));
   for(int i = 1; i < 256; ++i){
      //std::cout << hist.at<float>(i-1)<<endl;
      cumul_hist.at<float>(i) = cumul_hist.at<float>(i-1) + hist.at<float>(i-1);
   }
   
   /* Normalize */
   normalize(cumul_hist, cumul_hist, 0, 255, NORM_MINMAX, -1, Mat());
   cumul_hist.convertTo(cumul_hist, CV_8U);
   LUT(channel, cumul_hist, channel);
   
  }

Mat histogram_equalization(const Mat& img_in)
{
   // Write histogram equalization codes here
   CV_Assert(img_in.channels() == 3);

   Mat img_out = img_in.clone(); // Histogram equalization result
   
   /* Split three channels*/
   Mat channels[3];
   split(img_out, channels);
   
   /* Initiate frequencies of each channels */
   for(int i = 0; i < 3; ++i)
      /* PDF and CDF will be computed in a self-defined function:*/
      histogram_equalization_single_channel(channels[i]);

   /* Merge three channels*/
   merge(channels, 3, img_out);
   return img_out;
}

bool Question1(char* argv[])
{
   // Read in input images
   Mat input_image = imread(argv[2], IMREAD_COLOR);

   // Histogram equalization
   Mat output_image = histogram_equalization(input_image);

   // Write out the result
   string output_name = string(argv[3]) + string("output1.png");
   imwrite(output_name.c_str(), output_image);
   
   return true;
}

// ===================================================
// ===== Question 2: Frequency domain filtering ======
// ===================================================
void ftshift(Mat& src)
{
   
   /* ftshift: move low frequency part to the center*/
   int cx = src.cols/2;
   int cy = src.rows/2;
   
   Mat q0(src, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
   Mat q1(src, Rect(cx, 0, cx, cy));  // Top-Right
   Mat q2(src, Rect(0, cy, cx, cy));  // Bottom-Left
   Mat q3(src, Rect(cx, cy, cx, cy)); // Bottom-Right
   
   Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
   q0.copyTo(tmp);
   q3.copyTo(q0);
   tmp.copyTo(q3);
   
   q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
   q2.copyTo(q1);
   tmp.copyTo(q2);

}

Mat padding(Mat &img)
{
   /* Padding the image to get optimal size for FT */
   Mat padded = img;                            //expand input image to optimal size
   int m = getOptimalDFTSize( img.rows );
   int n = getOptimalDFTSize( img.cols ); // on the border add zero values
   copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));
   return padded;
}



Mat magnitude(Mat &complexI, Mat &dst)
{
   Mat channels[2];
   split(complexI, channels);
   magnitude(channels[0], channels[1], dst);
   //dst += Scalar::all(1);                    // switch to logarithmic scale
   //log(dst, dst);
   return dst;
}


Mat ft(Mat & src, Mat & dst)
{
   /* Compute FT */
   Mat planes[] = {Mat_<float>(src), Mat::zeros(src.size(), CV_32F)};
   merge(planes, 2, dst);         // Add to the expanded another plane with zeros
   normalize(planes[0], planes[0], 0, 1, NORM_MINMAX);
   dft(dst, dst);            // this way the result may fit in the source matrix
   ftshift(dst);
   return dst;
}

Mat ift(Mat & src, Mat & dst)
{
   
   Mat complexI = src.clone();
   ftshift(complexI);
   
   dft(complexI, complexI, DFT_INVERSE);
   magnitude(complexI,dst);
   return dst;
}


Mat low_pass_filter(const Mat& img_in)
{
   // Write low pass filter codes here
   Mat img_out = img_in.clone(); // Low pass filter result
   
   /* Padding the image to get optimal size for FT */
   Mat padded = padding(img_out);
   
   /* Compute FT */
   Mat complexI;
   ft(padded, complexI);
   
   /* Preseve low-frequency part only */
   
   Mat high_pass = complexI.clone();
   high_pass(Rect(padded.cols / 2 - 9, padded.rows / 2 - 9, 20, 20)).setTo(Scalar(0.,0.));
   complexI = complexI - high_pass;
   
   
   /* Back to spacial domain*/
   ift(complexI, img_out);
   normalize(img_out, img_out, 0, 255, NORM_MINMAX);
   img_out.convertTo(img_out, CV_8U);
   return img_out;
}

Mat high_pass_filter(const Mat& img_in)
{
   // Write high pass filter codes here
   Mat img_out = img_in.clone(); // High pass filter result

   
   /* Padding the image to get optimal size for FT */
   Mat padded = padding(img_out);

   
   /* Compute FT */
   Mat complexI;
   ft(padded, complexI);
   
   
   /* Remove low-frequency part only */
   complexI(Rect(padded.cols / 2 - 9, padded.rows / 2 - 9, 20, 20)).setTo(Scalar(0.,0.));
   
   /* Back to spacial domain*/
   ift(complexI, img_out);
   normalize(img_out, img_out, 0, 255, NORM_MINMAX);
   img_out.convertTo(img_out, CV_8U);
   return img_out;

}

Mat complex_division(Mat& Array1, Mat& Array2, Mat &dst)
{

   dst = Array1.clone();
   int width = Array1.cols;
   int height = Array1.rows;
   
   Mat_<Vec2f> kernal = Array2;
   Mat_<Vec2f> blurred = Array1;
   for (int y = 0; y < height; y++)
   {
      for (int x = 0; x < width; x++)
      {
         /* (a+bi) / (c + di) = 1 / (c ^ 2 + d ^ 2) * ((ac+ bd) + (bc - ad) i)*/
         /* Avoid that divider is zero */
         float power = kernal(y,x)[0] * kernal(y,x)[0] + kernal(y,x)[1] * kernal(y,x)[1];
         float factor = 1.f / power;
         
         float real = (blurred(y,x)[0] * kernal(y,x)[0] +  blurred(y,x)[1] * kernal(y,x)[1]) * factor;
         float img = (blurred(y,x)[1] * kernal(y,x)[0] -  blurred(y,x)[0] * kernal(y,x)[1]) * factor;
         dst.at<Vec2f>(y,x)[0] = real;
         dst.at<Vec2f>(y,x)[1] = img;
      }
   }
   return dst;

}

Mat deconvolution(const Mat& img_in)
{
   // Write deconvolution codes here
   Mat img_out = img_in.clone(); // Deconvolution result
   
   Mat gk = getGaussianKernel(21,5);
   gk = gk * gk.t();
  
   /* pad gk to match the size of img*/
   Mat gauss_mask;
   int w = img_out.cols-gk.cols;
   int h = img_out.rows-gk.rows;
   int r = w/2;
   int l = img_out.cols-gk.cols -r;
   int b = h/2;
   int t = img_out.rows-gk.rows -b;
   copyMakeBorder(gk,gauss_mask,t,b,l,r,BORDER_CONSTANT,Scalar(0,0));
   
  
   /* change input image and kernel to frequency domain*/
   Mat_<float> imf, gkf;
   ft(img_out, imf);
   ft(gauss_mask, gkf);
 

   /* Get estimation of f^ */
   Mat complexI;
   
   
   /* Do division between complex matrix element-wise: */
   complex_division(imf, gkf, complexI);
   
   /* Reconstruct picture */
   ift(complexI, img_out);
   
   ftshift(img_out);
   normalize(img_out, img_out, 0, 255, NORM_MINMAX);
   img_out.convertTo(img_out, CV_8U);
   return img_out;
}

bool Question2(char* argv[])
{
   // Read in input images
   Mat input_image1 = imread(argv[2], IMREAD_GRAYSCALE);
   Mat input_image2 = imread(argv[3], IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);

   // Low and high pass filters
   Mat output_image1 = low_pass_filter(input_image1);
   Mat output_image2 = high_pass_filter(input_image1);
   // Deconvolution
   Mat output_image3 = deconvolution(input_image2);

   // Write out the result
   string output_name1 = string(argv[4]) + string("output2LPF.png");
   string output_name2 = string(argv[4]) + string("output2HPF.png");
   string output_name3 = string(argv[4]) + string("output2deconv.png");
   imwrite(output_name1.c_str(), output_image1);
   imwrite(output_name2.c_str(), output_image2);
   imwrite(output_name3.c_str(), output_image3);

   return true;
}

// ===================================================
// ===== Question 3: Laplacian pyramid blending ======
// ===================================================

Mat laplacian_pyramid_blending(const Mat& img_in1, const Mat& img_in2)
{
   // Write laplacian pyramid blending codes here

   /* pyrDown first input */
   int ncols = img_in1.cols <  img_in2.cols ? img_in1.cols : img_in2.cols;
   int nrows = img_in1.rows <  img_in2.rows ? img_in1.rows : img_in2.rows;
   Mat in1 = img_in1(Rect(0,0,ncols, nrows));
   Mat in2 = img_in2(Rect(0,0,ncols, nrows));
   
   Mat gpA[6];
   Mat G = in1;
   gpA[0] = G;
   for(int i = 0; i < 5; ++i){
      pyrDown( gpA[i], G, Size( G.cols/2, G.rows/2) );
      gpA[i+1] = G;
   }
   
   /* pyrDown second input */
   Mat gpB[6];
   G = in2;
   gpB[0] = G;
   for(int i = 0; i < 5; ++i){
      pyrDown( gpB[i], G, Size( G.cols/2, G.rows/2 ) );
      gpB[i+1] = G;
   }
   
   /* generate Laplacian Pyramid for A */
   Mat lpA[6];
   lpA[0] = gpA[5];
   for (int i = 0; i < 5; ++i){
      Mat GE;
      pyrUp( gpA[5-i], GE, gpA[4-i].size());
      Mat L;
      subtract(gpA[4-i], GE, L);
      lpA[i+1] = L;
   }
   
   /* generate Laplacian Pyramid for B */
   Mat lpB[6];
   lpB[0] = gpB[5];
   for (int i = 0; i < 5; ++i){
      Mat GE;
      pyrUp( gpB[5-i], GE, gpB[4-i].size());
      Mat L;
      subtract(gpB[4-i], GE, L);
      lpB[i+1] = L;
   }
   
/* Now add left and right halves of images in each level*/
   Mat LS[6];
   for(int i = 0; i < 6; ++i){
      Mat Ls = Mat::zeros(lpA[i].size(), lpA[i].type());
      Mat la = lpA[i];
      Mat lb = lpB[i];
      la(Rect(0, 0, la.cols/2, la.rows)).copyTo(Ls(Rect(0, 0, la.cols/2, la.rows)));
      lb(Rect(lb.cols/2, 0, lb.cols/2, lb.rows)).copyTo(Ls(Rect(lb.cols/2, 0, lb.cols/2, lb.rows)));
      LS[i] = Ls;
   }
   
   /* now reconstruct */
   Mat ls_ = LS[0];
   for (int i = 1; i < 6; ++i){
      pyrUp( ls_, ls_, LS[i].size());
      add(ls_, LS[i], ls_);
   }
   
   Mat img_out = ls_;
   
   return img_out;
}

bool Question3(char* argv[])
{
   // Read in input images
   Mat input_image1 = imread(argv[2], IMREAD_COLOR);
   Mat input_image2 = imread(argv[3], IMREAD_COLOR);

   // Histogram equalization
   Mat output_image = laplacian_pyramid_blending(input_image1, input_image2);

   // Write out the result
   string output_name = string(argv[4]) + string("output3.png");
   imwrite(output_name.c_str(), output_image);

   return true;
}

int main(int argc, char* argv[])
{
   int question_number = -1;

   // Validate the input arguments
   if (argc < 4) {
      help_message(argv);
      exit(1);
   }
   else {
      question_number = atoi(argv[1]);

      if (question_number == 1 && !(argc == 4)) {
         help_message(argv);
         exit(1);
      }
      if (question_number == 2 && !(argc == 5)) {
	 help_message(argv);
	 exit(1);
      }
      if (question_number == 3 && !(argc == 5)) {
	 help_message(argv);
	 exit(1);
      }
      if (question_number > 3 || question_number < 1 || argc > 5) {
	 cout << "Input parameters out of bound ..." << endl;
	 exit(1);
      }
   }

   switch (question_number) {
      case 1: Question1(argv); break;
      case 2: Question2(argv); break;
      case 3: Question3(argv); break;
   }

   return 0;
}
