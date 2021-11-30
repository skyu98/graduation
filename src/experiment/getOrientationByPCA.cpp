void Jar::getOrientation() {
     /* 最大轮廓minRect方向 */
    RotatedRect rrect = minAreaRect(*originalContour);
    if(rrect.size.width < rrect.size.height) {
        rrect.angle += 90;
    }

    /* PCA */
    // Construct a buffer used by the pca analysis
    size_t pointCount = originalContour->size(); 
    Mat pca_data = Mat(static_cast<int>(pointCount), 2, CV_64FC1); // n rows * 2 cols(x, y)

    for(size_t i = 0;i < pointCount;++i) {
        pca_data.at<double>(i, 0) = (*originalContour)[i].x;
        pca_data.at<double>(i, 1) = (*originalContour)[i].y;
    }

    // Perform PCA
    cv::PCA pca_analysis(pca_data, Mat(), CV_PCA_DATA_AS_ROW);

    posture.center.x = pca_analysis.mean.at<double>(0, 0); 
    posture.center.y = pca_analysis.mean.at<double>(0, 1); 
    
    // 2 eigenvectors/eigenvalues are enough
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);

    for (size_t i = 0; i < 2; ++i) {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i,0);
    }

    // Get the eigenvec angle, range: (-pi, pi]
    double eigenvec_angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x);
    double eigenvec_angle_double = 180 * (eigenvec_angle) / M_PI;

    if(eigenvec_angle_double > 90) {
        eigenvec_angle_double -= 180;
    }
    else if(eigenvec_angle_double < -1 * 90) {
        eigenvec_angle_double += 180;
    }


    posture.angle_double = rrect.angle;
    posture.angle = M_PI * posture.angle_double / 180;
    line(*paintImg, posture.center, posture.center + 800 * Point2d(cos(posture.angle), sin(posture.angle)) , CV_RGB(0, 255, 0));
    
    posture.angle_double = eigenvec_angle_double;
    posture.angle = M_PI * posture.angle_double / 180;
    line(*paintImg, posture.center, posture.center + 800 * Point2d(cos(posture.angle), sin(posture.angle)) , CV_RGB(255, 0, 0));

    posture.angle_double = rrect.angle;
    posture.angle_double += 7 * eigenvec_angle_double; 
    posture.angle_double /= 8; 
    posture.angle = M_PI * posture.angle_double / 180;

    // Draw the principal components
    // 在轮廓中点绘制小圆
    circle(*paintImg, posture.center, 3, CV_RGB(255, 0, 255), 2);
    //计算出直线，在主要方向上绘制直线
    line(*paintImg, posture.center, posture.center + 800 * Point2d(cos(posture.angle), sin(posture.angle)) , CV_RGB(255, 255, 0));
}