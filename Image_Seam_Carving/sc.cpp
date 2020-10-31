
#include "sc.h"
#include <limits.h>
#include <algorithm>

using namespace cv;
using namespace std;

//declaring Mat to store enery value of each pixel of the input image
Mat energy_image;
Mat gray; // for storing the converted color image
Mat grad; // to store the input image after applying the sobel function
Mat grad_x,grad_y;
Mat abs_grad_x,abs_grad_y;
int scale = 1;
int delta = 0;
int ddepth = CV_64F;

int HEIGHT;
int WIDTH;
int kernel_size = 3;


bool seam_carving(Mat& in_image, int new_width, int new_height, Mat& out_image){

    // some sanity checks
    // Check 1 -> new_width <= in_image.cols
    if(new_width>in_image.cols){
        cout<<"Invalid request!!! new_width has to be smaller than the current size!"<<endl;
        return false;
    }
    if(new_height>in_image.rows){
        cout<<"Invalid request!!! ne_height has to be smaller than the current size!"<<endl;
        return false;
    }

    if(new_width<=0){
        cout<<"Invalid request!!! new_width has to be positive!"<<endl;
        return false;

    }

    if(new_height<=0){
        cout<<"Invalid request!!! new_height has to be positive!"<<endl;
        return false;

    }

    return seam_carving_trivial(in_image, new_width, new_height, out_image);
}


// seam carves by removing trivial seams
bool seam_carving_trivial(Mat& in_image, int new_width, int new_height, Mat& out_image){

    Mat iimage = in_image.clone();
    Mat oimage = in_image.clone();

    
    while(iimage.rows!=new_height || iimage.cols!=new_width){
        GaussianBlur(iimage,iimage,Size(3,3),0,0,BORDER_DEFAULT);
        cvtColor(iimage,gray,CV_BGR2GRAY);
        //Laplacian(iimage,grad_x,ddepth,kernel_size,scale,delta,BORDER_DEFAULT);
        //convertScaleAbs(grad_x,grad);
        Scharr(gray,grad_x,ddepth,1,0,scale,delta,BORDER_DEFAULT);    //Sobel
        Scharr(gray,grad_y,ddepth,0,1,scale,delta,BORDER_DEFAULT);    //Sobel
        //Sobel(gray,grad_x,ddepth,1,0,3,scale,delta,BORDER_DEFAULT);
        //Sobel(gray,grad_y,ddepth,0,1,3,scale,delta,BORDER_DEFAULT);
        convertScaleAbs(grad_x,abs_grad_x);
        convertScaleAbs(grad_y,abs_grad_y);
        addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0,grad);
        //addWeighted(grad_x,0.5,grad_y,0.5,0,grad);
        //imshow("IMAGE AFTER ENERGY FUNCTION",grad);
        // horizontal seam if needed
        /*
        for(int y=0;y<grad.rows;y++){
            for(int x=0;x<grad.cols;x++){
                
                cout << grad.at<short int>(y,x) << "," ;
            }
            cout << endl;
            
        }
        */
        
        if(iimage.rows>new_height){
            
            reduce_horizontal_seam_trivial(iimage, oimage);
           
            iimage = oimage.clone();
             //cout << "iimage size:" << iimage.size() << endl;
        }
        
        
        GaussianBlur(iimage,iimage,Size(3,3),0,0,BORDER_DEFAULT);
        cvtColor(iimage,gray,CV_BGR2GRAY);
        //Laplacian(iimage,grad_x,ddepth,kernel_size,scale,delta,BORDER_DEFAULT);
        //convertScaleAbs(grad_x,grad);
        //Scharr(gray,grad_x,ddepth,1,0,scale,delta,BORDER_DEFAULT);    //Sobel
        //Scharr(gray,grad_y,ddepth,0,1,scale,delta,BORDER_DEFAULT);    //Sobel
        Sobel(gray,grad_x,ddepth,1,0,3,scale,delta,BORDER_DEFAULT);
        Sobel(gray,grad_y,ddepth,0,1,3,scale,delta,BORDER_DEFAULT);
        convertScaleAbs(grad_x,abs_grad_x);
        convertScaleAbs(grad_y,abs_grad_y);
        addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0,grad);
        //addWeighted(grad_x,0.5,grad_y,0.5,0,grad);
        //imshow("IMAGE AFTER ENERGY FUNCTION",grad);
        //cout <<
        if(iimage.cols>new_width){
            
            reduce_vertical_seam_trivial(iimage, oimage);
            iimage = oimage.clone();
        }
    }

    out_image = oimage.clone();
    return true;
}

// horizontl trivial seam is a seam through the center of the image
bool reduce_horizontal_seam_trivial(Mat& in_image, Mat& out_image){
    Mat Energy(in_image.rows,in_image.cols, CV_32SC(1), Scalar::all(1));
    
    for(int y=0;y<Energy.rows;y++)
        for(int x=0;x<Energy.cols;x++)
        Energy.at<int>(y,x) = (int)grad.at<uchar>(y,x);
    
    
    vector<int> adjacentPixel;
    vector<int> finalEnergy;
    
    int max = INT_MAX;
    for(int x=in_image.cols-2;x>-1;x--){
        
        for(int y=0;y<in_image.rows;y++){
            
            adjacentPixel.clear();
            if(y==0){
                adjacentPixel.push_back(INT_MAX);
            }
            else{
                adjacentPixel.push_back(Energy.at<int>(y-1,x+1));
            }
            
            adjacentPixel.push_back(Energy.at<int>(y,x+1));
         
            if(y==in_image.rows-1){
                adjacentPixel.push_back(INT_MAX);
            }
            else{
                adjacentPixel.push_back(Energy.at<int>(y+1,x+1));
              
            }
            
            int direction = 0;
            for(int i=0;i<adjacentPixel.size();i++){
            if(adjacentPixel[i]<max){
                    max=adjacentPixel[i];
                    direction = i;
            }
            }
            max = INT_MAX;
            
            if(direction == 0){
                Energy.at<int>(y,x) += Energy.at<int>(y-1,x+1);
            
            }
            
            else if(direction == 1){
                Energy.at<int>(y,x) += Energy.at<int>(y,x+1);
            }
            
            else if(direction == 2){
               Energy.at<int>(y,x) += Energy.at<int>(y+1,x+1);
            }
            
            
            if(x==0){
             
                finalEnergy.push_back(Energy.at<int>(y,x));
            }
            
            
        }
        
        
    }
    
    //cout << "yes" << endl;
    int chosenRow = std::min_element(finalEnergy.begin(),finalEnergy.end()) - finalEnergy.begin();
    //cout << "chosen row:" << chosenRow << endl;
    int minEnergy = *std::min_element(finalEnergy.begin(), finalEnergy.end());
    //cout << "seam value:" << Energy.at<int>(chosenRow,0) << endl;
    
    vector<int> fdirection;
    int ych = chosenRow;
    adjacentPixel.clear();
    Vec3b red(0,0,255);
    out_image.at<Vec3b>(ych,0) = red;
    
    for(int x=0;x<in_image.cols-1;x++){
        
        adjacentPixel.clear();
        if(ych==0){
            adjacentPixel.push_back(INT_MAX);
        }
        else{
            adjacentPixel.push_back(Energy.at<int>(ych-1,x+1));
        }
            
        adjacentPixel.push_back(Energy.at<int>(ych,x+1));
            
        if(ych==in_image.rows-1){
            adjacentPixel.push_back(INT_MAX);
        }
        else{
            adjacentPixel.push_back(Energy.at<int>(ych+1,x+1));
        }
            
        int direction = 0;
        for(int i=0;i<adjacentPixel.size();i++){
            if(adjacentPixel[i]<max){
                max=adjacentPixel[i];
                direction = i;
        }
        }
        
        max = INT_MAX;
        if(direction==0){
            ych=ych-1;
            out_image.at<Vec3b>(ych,x+1) = red;
            
        }
        else if(direction == 1){
            ych = ych;
            out_image.at<Vec3b>(ych,x+1) = red;
            
        }
        else if(direction == 2){
            ych = ych+1;
            out_image.at<Vec3b>(ych,x+1) = red;
            
        }
        fdirection.push_back(direction);
    }
    
    //cout << "direction size:" << fdirection.size() << endl;
    
    
    vector<int> visited(in_image.cols,0);
    Mat finalOutput = Mat(in_image.rows-1,in_image.cols,CV_8UC3);
           // cout << "size of final output:" << finalOutput.rows << "," << finalOutput.cols << endl;
        for(int y=0;y<finalOutput.rows;y++){
            for(int x=0;x<finalOutput.cols;x++){

                    if(visited[x]==0){
                finalOutput.at<Vec3b>(y,x)=out_image.at<Vec3b>(y,x);
                    }
                    if(out_image.at<Vec3b>(y,x)==red){
                visited[x]=1;
                        
                            finalOutput.at<Vec3b>(y,x)=out_image.at<Vec3b>(y+1,x);
                        
                    }
                    else if(visited[x]==1){
                        
                            finalOutput.at<Vec3b>(y,x)=out_image.at<Vec3b>(y+1,x);
                    }
            }
        }
    
    //imshow("IMAGE AFTER SEAM MARKING",out_image);
   
    out_image = finalOutput.clone();
    return true;
}


// vertical trivial seam is a seam through the center of the image
bool reduce_vertical_seam_trivial(Mat& in_image, Mat& out_image){
    Mat Energy(in_image.rows,in_image.cols, CV_32SC(1), Scalar::all(0));
    for(int y=0;y<Energy.rows;y++)
        for(int x=0;x<Energy.cols;x++)
        Energy.at<int>(y,x) = (int)grad.at<uchar>(y,x);
    
    //cout << "Grad:" << (int)grad.at<uchar>(10,10) << endl;
    //cout << "Energy:" << Energy.at<int>(10,10) << endl;
    
    //cout << "Energy 0,1397:" << Energy.at<int>(0,1397) << endl;
    vector<int> adjacentPixel;
    vector<int> finalEnergy;
    
    int max = INT_MAX;
    for(int y=in_image.rows-2;y>-1;y--){
        
        for(int x=0;x<in_image.cols;x++){
            
            adjacentPixel.clear();
            if(x==0){
                adjacentPixel.push_back(INT_MAX);
            }
            else{
                adjacentPixel.push_back(Energy.at<int>(y+1,x-1));
            }
            
            adjacentPixel.push_back(Energy.at<int>(y+1,x));
         
            if(x==in_image.cols-1){
                adjacentPixel.push_back(INT_MAX);
            }
            else{
                adjacentPixel.push_back(Energy.at<int>(y+1,x+1));
              
            }
            int direction = std::min_element(adjacentPixel.begin(),adjacentPixel.end()) - adjacentPixel.begin();
            //int direction = 0;
            /*
            for(int i=0;i<adjacentPixel.size();i++){
            if(adjacentPixel[i]<max){
                    max=adjacentPixel[i];
                    direction = i;
            }
            }
            */
            max = INT_MAX;
            
            if(direction == 0){
                Energy.at<int>(y,x) += Energy.at<int>(y+1,x-1);
            
            }
            
            else if(direction == 1){
                Energy.at<int>(y,x) += Energy.at<int>(y+1,x);
            }
            
            else if(direction == 2){
               Energy.at<int>(y,x) += Energy.at<int>(y+1,x+1);
            }
            
            
            if(y==0){
             
                finalEnergy.push_back(Energy.at<int>(y,x));
            }
            
            
        }
        
        
    }
    /*
    for(int y=0;y<Energy.rows;y++){
        for(int x=0;x<Energy.cols;x++){
            cout << Energy.at<int>(y,x) << " ";
        }
        cout << endl;
    }
    */
    int chosenCol = std::min_element(finalEnergy.begin(),finalEnergy.end()) - finalEnergy.begin();
    //cout << "chosen column:" << chosenCol << endl;
    int minEnergy = *std::min_element(finalEnergy.begin(), finalEnergy.end());
    //cout << "seam value:" << Energy.at<int>(0,chosenCol) << endl;
    
    //cout << Energy.at<int>(1,chosenCol-1) << endl;
   // cout << Energy.at<int>(1,chosenCol) << endl;
   // cout << Energy.at<int>(1,chosenCol+1) << endl;
    //cout << Energy.at<int>(1,choseCol-1) << endl;
    vector<int> fdirection;
    int xch = chosenCol;
    adjacentPixel.clear();
    Vec3b red(0,0,255);
    out_image.at<Vec3b>(0,xch) = red;
    
    //cout << "Direction:" << endl;
    for(int y=0;y<in_image.rows-1;y++){
        
        adjacentPixel.clear();
        if(xch==0){
            adjacentPixel.push_back(INT_MAX);
        }
        else{
            adjacentPixel.push_back(Energy.at<int>(y+1,xch-1));
        }
            
        adjacentPixel.push_back(Energy.at<int>(y+1,xch));
            
        if(xch==in_image.cols-1){
            adjacentPixel.push_back(INT_MAX);
        }
        else{
            adjacentPixel.push_back(Energy.at<int>(y+1,xch+1));
        }
            
        int direction = std::min_element(adjacentPixel.begin(),adjacentPixel.end()) - adjacentPixel.begin();
            
        max = INT_MAX;
        if(direction==0){
            xch=xch-1;
            out_image.at<Vec3b>(y+1,xch) = red;
            
        }
        else if(direction == 1){
            xch = xch;
            out_image.at<Vec3b>(y+1,xch) = red;
            
        }
        else if(direction == 2){
            xch = xch+1;
            out_image.at<Vec3b>(y+1,xch) = red;
            
        }
        //fdirection.push_back(direction);
    }
    
    //cout << "direction size:" << fdirection.size() << endl;
    
    
    vector<int> visited(in_image.rows,0);
    Mat finalOutput = Mat(in_image.rows,in_image.cols-1,CV_8UC3);
            //cout << "size of final output:" << finalOutput.rows << "," << finalOutput.cols << endl;
        for(int x=0;x<finalOutput.cols;x++){
            for(int y=0;y<finalOutput.rows;y++){

                    if(visited[y]==0){
                finalOutput.at<Vec3b>(y,x)=out_image.at<Vec3b>(y,x);
                    }
                    if(out_image.at<Vec3b>(y,x)==red){
                visited[y]=1;
                        
                            finalOutput.at<Vec3b>(y,x)=out_image.at<Vec3b>(y,x+1);
                    }
                    else if(visited[y]==1){
                
                            finalOutput.at<Vec3b>(y,x)=out_image.at<Vec3b>(y,x+1);
                    }
            }
        }
    
    //imshow("IMAGE AFTER SEAM MARKING",out_image);
   
    out_image = finalOutput.clone();
    return true;
}
