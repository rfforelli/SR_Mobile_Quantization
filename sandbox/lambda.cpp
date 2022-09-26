#include <iostream>
#include <vector>
#include <assert.h>

///////////////////////////////////////////////////////////
//////helper functions for debugging, not for synthesis////
///////////////////////////////////////////////////////////
void helper_sum_img (std::vector<std::vector<std::vector<std::vector<int>>>> img){
    //get incoming image dimensions
    int N = img.size();
    int H = img[0].size();
    int W = img[0][0].size();
    int C = img[0][0][0].size();
    

    std::vector<std::vector<std::vector<std::vector<int>>>> img_out(N,std::vector<std::vector<std::vector<int>>>(H,(std::vector<std::vector<int>>(W, std::vector<int>(C, 0)))));
    int sum = 0;
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < C; c++) {
                    sum =  sum+ img[n][h][w][c];
                }
            }
        }
    }
    std::cout << "Sum of image is: " << sum;
}

void helper_print_4x4 (std::vector<std::vector<std::vector<std::vector<int>>>> img){

    int N = img.size();
    int H = img[0].size();
    int W = img[0][0].size();
    int C = img[0][0][0].size();
    assert(C > 4 && W > 4);
    std::cout << N;
    std::cout << "\n";
    std::cout << H;
    std::cout << "\n";
    std::cout << W;
    std::cout << "\n";
    std::cout << C;
    std::cout << "\n";

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            std::cout << img[0][0][i][j];
            std::cout << " ";
        }
        std::cout << "\n";
    }
}

std::vector<std::vector<std::vector<std::vector<int>>>> identity_img (std::vector<std::vector<std::vector<std::vector<int>>>> img){
    return img;
}
//////helper functions for debugging, not for synthesis////
///////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////
/////////////////////LAMBDA LAYERS/////////////////////////

std::vector<std::vector<std::vector<std::vector<int>>>> upsample_img(std::vector<std::vector<std::vector<std::vector<int>>>> img, int scale){

   int N = img.size();
   int H = img[0].size();
   int W = img[0][0].size();
   int C = img[0][0][0].size();
   int new_C = C * scale * scale;

   std::vector<std::vector<std::vector<std::vector<int>>>> upsampled_img(N,std::vector<std::vector<std::vector<int>>>(H,(std::vector<std::vector<int>>(W, std::vector<int>(new_C, 0)))));
   for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < new_C; c++) {
                    upsampled_img[n][h][w][c] = img[n][h][w][c % C];
                }
            }
        }
    }
    return upsampled_img;
}

std::vector<std::vector<std::vector<std::vector<int>>>> depth_to_space(std::vector<std::vector<std::vector<std::vector<int>>>> img, int block_size){

	// N, H, W, C = img.shape
    int N = img.size();
    int H = img[0].size();
    int W = img[0][0].size();
    int C = img[0][0][0].size();
    
    int new_H = H*block_size;
    int new_W = W*block_size;
    int new_C = C/(block_size*block_size);
    assert (block_size*block_size <= C);
    std::vector<int> tmp_list(N*H*W*C);


    std::vector<std::vector<std::vector<std::vector<int>>>> img_out(N,std::vector<std::vector<std::vector<int>>>(new_H,(std::vector<std::vector<int>>(new_W, std::vector<int>(new_C, 0)))));
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < C; c++) {
                    for (int x =0; x< block_size; x++){
                        for (int y = 0; x < block_size; x++) {
                            if (w+x < W && c + y < C) {
                                tmp_list.push_back(img[n][h][w+x][c+y]);
                            }
                        }
                    }
                }
            }
        }
    }

    //remap the intermediate list with the correct indices
    //remap the first 216
    for (int counter=0; counter <216; counter++){
        for (auto it = img_out.begin(); it != img_out.end(); ++it){
            for (auto it1 = it->begin(); it1 != it->end(); ++it1){
                for (auto it2 = it1->begin(); it2 != it1->end(); ++it2){
                    for (auto it3 = it2->begin(); it3 != it2->end(); ++it3){
                        *it3 = tmp_list[counter*3]; 
                    }
                }
            }
        }
    }

    //remap the next 216
    for (int counter=216; counter <432; counter++){
        for (auto it = img_out.begin(); it != img_out.end(); ++it){
            for (auto it1 = it->begin(); it1 != it->end(); ++it1){
                for (auto it2 = it1->begin(); it2 != it1->end(); ++it2){
                    for (auto it3 = it2->begin(); it3 != it2->end(); ++it3){
                        *it3 = tmp_list[(counter-216)*3+1]; 
                    }
                }
            }
        }
    }

    //remap the next 216
    for (int counter=432; counter <648; counter++){
        for (auto it = img_out.begin(); it != img_out.end(); ++it){
            for (auto it1 = it->begin(); it1 != it->end(); ++it1){
                for (auto it2 = it1->begin(); it2 != it1->end(); ++it2){
                    for (auto it3 = it2->begin(); it3 != it2->end(); ++it3){
                        *it3 = tmp_list[(counter-432)*3+2]; 
                    }
                }
            }
        }
    }

    return img_out;
}

std::vector<std::vector<std::vector<std::vector<int>>>> add_img (std::vector<std::vector<std::vector<std::vector<int>>>> img_1, std::vector<std::vector<std::vector<std::vector<int>>>> img_2){
    //get incoming image dimensions
    int N = img_1.size();
    int H = img_1[0].size();
    int W = img_1[0][0].size();
    int C = img_1[0][0][0].size();	
    

    std::vector<std::vector<std::vector<std::vector<int>>>> img_out(N,std::vector<std::vector<std::vector<int>>>(H,(std::vector<std::vector<int>>(W, std::vector<int>(C, 0)))));
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < C; c++) {
                    img_out[n][h][w][c] = img_1[n][h][w][c] + img_2[n][h][w][c];
                }
            }
        }
    }
    return img_out;
}

std::vector<std::vector<std::vector<std::vector<int>>>> clip_img (std::vector<std::vector<std::vector<std::vector<int>>>> img, int lower, int upper){
    //get incoming image dimensions
    int N = img.size();
    int H = img[0].size();
    int W = img[0][0].size();
    int C = img[0][0][0].size();

    //set outgoing image dimensions
    // int img_out[N][H][W][C];
    std::vector<std::vector<std::vector<std::vector<int>>>> img_out(N,std::vector<std::vector<std::vector<int>>>(H,(std::vector<std::vector<int>>(W, std::vector<int>(C, 0)))));
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < C; c++) {
                    img_out[n][h][w][c] = std::max(lower, std::min(upper, img[n][h][w][c]));
                }
            }
        }
    }
    return img_out;
}

int main() {
    int N = 1, H = 40, W = 4, C = 4;
    std::vector<std::vector<std::vector<std::vector<int>>>> img(N,std::vector<std::vector<std::vector<int>>>(H,(std::vector<std::vector<int>>(W, std::vector<int>(C, 200)))));
    // print_4x4(img);
    // print_4x4(upsample_img(img, 3));
    helper_print_4x4(add_img(img,img));

}