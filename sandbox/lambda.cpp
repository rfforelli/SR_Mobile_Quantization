int* upsample_img(int* img, int scale){
   
   //takes in 5d vector: [num_vectors][N][H][W][C]
   int num_vectors = sizeof(img)/sizeof(a[0]);
   int N = sizeof(img[0])/sizeof(img[0][0]);
   int H = sizeof(img[0][0])/sizeof(img[0][0][0]);
   int W = sizeof(img[0][0][0])/sizeof(img[0][0][0][0]);
   int C = sizeof(img[0][0][0][0])/sizeof(img[0][0])[0][0][0];
   int img_out = [N][H][W][C*num_vectors]
   for (v = 0; v < num_vectors; v++) {
      for (n = 0; n < N; n++) {
         for (h = 0; h < H; h++) {
            for (w = 0; w < W; w++) {
                for (c = 0; c < C; c++) {
                    img_out[n][h][w][v*C + c] = img[v][n][h][w][c]
                }
            }
         }
      } 
   }  
}

int** depth_to_space(int** img, block_size){

	N, H, W, C = img.shape
	assert(C % (block_size*block_size) == 0)
	}

int* add_img (int* img_1, int* img_2){
    //get incoming image dimensions
    int N = sizeof(img_1)/sizeof(a[img_1]);
    int H = sizeof(img_1[0])/sizeof(img_1[0][0]);
    int W = sizeof(img_1[0][0])/sizeof(img_1[0][0][0]);
    int C = sizeof(img_1[0][0][0])/sizeof(img_1[0][0])[0][0];	
    
    //set outgoing image dimensions
    int img_out[N][H][W][C];
    for (n = 0; n < N; n++) {
        for (h = 0; h < H; h++) {
            for (w = 0; w < W; w++) {
                for (c = 0; c < C; c++) {
                    img_out[n][h][w][c] = img_1[n][h][w][c] + img_2[n][h][w][c]
                }
            }
        }
    }
    return img_out
}

int* clip_img (int* img, int min, int max){
    //get incoming image dimensions
    int N = sizeof(img)/sizeof(img[0]);
    int H = sizeof(img[0])/sizeof(img[0][0]);
    int W = sizeof(img[0][0])/sizeof(img[0][0][0]);
    int C = sizeof(img[0][0][0])/sizeof(img[0][0][0][0]);

    //set outgoing image dimensions
    int img_out[N][H][W][C];
    for (n = 0; n < N; n++) {
        for (h = 0; h < H; h++) {
            for (w = 0; w < W; w++) {
                for (c = 0; c < C; c++) {
                    img_out[n][h][w][c] = std::max(lower, std::min(upper, img[n][h][w][c])
                }
            }
        }
    }
    return img_out
}