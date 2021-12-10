#include "imageCropper.h"

torch::NoGradGuard imgCropper::no_grad;

int imgCropper::init(const string& cfgPath, const string& weightsPath, int input_image_size) {
    net_ = new Darknet(cfgPath.c_str(), &device_);

    map<string, string> *info = net_->get_net_info();
    info->operator[]("height") = std::to_string(input_image_size);

    std::cout << "Loading weight ..." << endl;
    net_->load_weights(weightsPath.c_str());
    std::cout << "Weight loaded ..." << endl;

    net_->to(device_);
    net_->eval();
}


cv::Rect imgCropper::getCroppedBox(const Mat& src, int gap) {
    Mat resized_image;
    cv::cvtColor(src, resized_image, cv::COLOR_BGR2RGB);

    int input_image_size = atoi(net_->get_net_info()->operator[]("height").c_str());
    cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0/255);

    auto img_tensor = torch::from_blob(img_float.data, {1, input_image_size,
                                            input_image_size, 3}).to(device_);
    img_tensor = img_tensor.permute({0,3,1,2});

    auto start = std::chrono::high_resolution_clock::now();
       
    auto output = net_->forward(img_tensor);
    
    // filter result by NMS 
    // class_num = 1
    // confidence = 0.2
    auto result = net_->write_results(output, 1, 0.2, 0.5);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); 

    // It should be known that it takes longer time at first time
    std::cout << "Inference taken : " << duration.count() << " ms" << endl; 

    if (result.dim() == 1) {
        std::cout << "No object found" << endl;
        return cv::Rect();
    }
    else {
        int obj_num = result.size(0);

        // std::cout << obj_num << " objects found" << endl;

        float w_scale = float(src.cols) / input_image_size;
        float h_scale = float(src.rows) / input_image_size;

        result.select(1,1).mul_(w_scale);
        result.select(1,2).mul_(h_scale);
        result.select(1,3).mul_(w_scale);
        result.select(1,4).mul_(h_scale);

        auto result_data = result.accessor<float, 2>();

        topLeft_ = cv::Point(result_data[0][1], result_data[0][2]);
        bottomRight_ = cv::Point(result_data[0][3], result_data[0][4]);

        topLeft_.x = max(topLeft_.x - gap, 0);
        topLeft_.y = max(topLeft_.y - gap, 0);

        bottomRight_.x = min(bottomRight_.x + gap , src.cols);
        bottomRight_.y = min(bottomRight_.y + gap , src.rows);

        int width = bottomRight_.x - topLeft_.x, height = bottomRight_.y - topLeft_.y;
        
        assert(width >= 0 && height >= 0);

        // std::cout << "Done" << endl;
        return {topLeft_.x, topLeft_.y, width, height};
        // for (int i = 0; i < result.size(0) ; i++)
        // {
        //     cv::rectangle(src, cv::Point(result_data[i][1], result_data[i][2]), cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(0, 0, 255), 1, 1, 0);
        // }

        // cv::imwrite("out-det.jpg", src);
    }
}