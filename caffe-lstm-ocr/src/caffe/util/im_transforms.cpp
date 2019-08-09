#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>

#if CV_VERSION_MAJOR == 3
#include <opencv2/imgcodecs/imgcodecs.hpp>
#define CV_GRAY2BGR cv::COLOR_GRAY2BGR
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#define CV_BGR2YCrCb cv::COLOR_BGR2YCrCb
#define CV_YCrCb2BGR cv::COLOR_YCrCb2BGR
#define CV_IMWRITE_JPEG_QUALITY cv::IMWRITE_JPEG_QUALITY
#define CV_LOAD_IMAGE_COLOR cv::IMREAD_COLOR
#define CV_THRESH_BINARY_INV cv::THRESH_BINARY_INV
#define CV_THRESH_OTSU cv::THRESH_OTSU
#endif
#endif  // USE_OPENCV

#include <algorithm>
#include <numeric>
#include <vector>

#include "caffe/util/im_transforms.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

const float prob_eps = 0.01;

int roll_weighted_die(const vector<float>& probabilities) {
  vector<float> cumulative;
  std::partial_sum(&probabilities[0], &probabilities[0] + probabilities.size(),
                   std::back_inserter(cumulative));
  float val;
  caffe_rng_uniform(1, static_cast<float>(0), cumulative.back(), &val);

  // Find the position within the sequence and add 1
  return (std::lower_bound(cumulative.begin(), cumulative.end(), val)
          - cumulative.begin());
}

void UpdateBBoxByResizePolicy(const ResizeParameter& param,
                              const int old_width, const int old_height,
                              NormalizedBBox* bbox) {
  float new_height = param.height();
  float new_width = param.width();
  float orig_aspect = static_cast<float>(old_width) / old_height;
  float new_aspect = new_width / new_height;

  float x_min = bbox->xmin() * old_width;
  float y_min = bbox->ymin() * old_height;
  float x_max = bbox->xmax() * old_width;
  float y_max = bbox->ymax() * old_height;
  float padding;
  switch (param.resize_mode()) {
    case ResizeParameter_Resize_mode_WARP:
      x_min = std::max(0.f, x_min * new_width / old_width);
      x_max = std::min(new_width, x_max * new_width / old_width);
      y_min = std::max(0.f, y_min * new_height / old_height);
      y_max = std::min(new_height, y_max * new_height / old_height);
      break;
    case ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD:
      if (orig_aspect > new_aspect) {
        padding = (new_height - new_width / orig_aspect) / 2;
        x_min = std::max(0.f, x_min * new_width / old_width);
        x_max = std::min(new_width, x_max * new_width / old_width);
        y_min = y_min * (new_height - 2 * padding) / old_height;
        y_min = padding + std::max(0.f, y_min);
        y_max = y_max * (new_height - 2 * padding) / old_height;
        y_max = padding + std::min(new_height, y_max);
      } else {
        padding = (new_width - orig_aspect * new_height) / 2;
        x_min = x_min * (new_width - 2 * padding) / old_width;
        x_min = padding + std::max(0.f, x_min);
        x_max = x_max * (new_width - 2 * padding) / old_width;
        x_max = padding + std::min(new_width, x_max);
        y_min = std::max(0.f, y_min * new_height / old_height);
        y_max = std::min(new_height, y_max * new_height / old_height);
      }
      break;
    case ResizeParameter_Resize_mode_FIT_SMALL_SIZE:
      if (orig_aspect < new_aspect) {
        new_height = new_width / orig_aspect;
      } else {
        new_width = orig_aspect * new_height;
      }
      x_min = std::max(0.f, x_min * new_width / old_width);
      x_max = std::min(new_width, x_max * new_width / old_width);
      y_min = std::max(0.f, y_min * new_height / old_height);
      y_max = std::min(new_height, y_max * new_height / old_height);
      break;
    default:
      LOG(FATAL) << "Unknown resize mode.";
  }
  bbox->set_xmin(x_min / new_width);
  bbox->set_ymin(y_min / new_height);
  bbox->set_xmax(x_max / new_width);
  bbox->set_ymax(y_max / new_height);
}

void InferNewSize(const ResizeParameter& resize_param,
                  const int old_width, const int old_height,
                  int* new_width, int* new_height) {
  int height = resize_param.height();
  int width = resize_param.width();
  float orig_aspect = static_cast<float>(old_width) / old_height;
  float aspect = static_cast<float>(width) / height;

  switch (resize_param.resize_mode()) {
    case ResizeParameter_Resize_mode_WARP:
      break;
    case ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD:
      break;
    case ResizeParameter_Resize_mode_FIT_SMALL_SIZE:
      if (orig_aspect < aspect) {
        height = static_cast<int>(width / orig_aspect);
      } else {
        width = static_cast<int>(orig_aspect * height);
      }
      break;
    case ResizeParameter_Resize_mode_FIT_HEIGHT_AND_PAD:
      break;
    case ResizeParameter_Resize_mode_FIT_HEIGHT_AND_RANDOM_PAD:
      break;
    default:
      LOG(FATAL) << "Unknown resize mode.";
  }
  *new_height = height;
  *new_width = width;
}

#ifdef USE_OPENCV
template <typename T>
bool is_border(const cv::Mat& edge, T color) {
  cv::Mat im = edge.clone().reshape(0, 1);
  bool res = true;
  for (int i = 0; i < im.cols; ++i) {
    res &= (color == im.at<T>(0, i));
  }
  return res;
}

template
bool is_border(const cv::Mat& edge, uchar color);

template <typename T>
cv::Rect CropMask(const cv::Mat& src, T point, int padding) {
  cv::Rect win(0, 0, src.cols, src.rows);

  vector<cv::Rect> edges;
  edges.push_back(cv::Rect(0, 0, src.cols, 1));
  edges.push_back(cv::Rect(src.cols-2, 0, 1, src.rows));
  edges.push_back(cv::Rect(0, src.rows-2, src.cols, 1));
  edges.push_back(cv::Rect(0, 0, 1, src.rows));

  cv::Mat edge;
  int nborder = 0;
  T color = src.at<T>(0, 0);
  for (int i = 0; i < edges.size(); ++i) {
    edge = src(edges[i]);
    nborder += is_border(edge, color);
  }

  if (nborder < 4) {
    return win;
  }

  bool next;
  do {
    edge = src(cv::Rect(win.x, win.height - 2, win.width, 1));
    next = is_border(edge, color);
    if (next) {
      win.height--;
    }
  } while (next && (win.height > 0));

  do {
    edge = src(cv::Rect(win.width - 2, win.y, 1, win.height));
    next = is_border(edge, color);
    if (next) {
      win.width--;
    }
  } while (next && (win.width > 0));

  do {
    edge = src(cv::Rect(win.x, win.y, win.width, 1));
    next = is_border(edge, color);
    if (next) {
      win.y++;
      win.height--;
    }
  } while (next && (win.y <= src.rows));

  do {
    edge = src(cv::Rect(win.x, win.y, 1, win.height));
    next = is_border(edge, color);
    if (next) {
      win.x++;
      win.width--;
    }
  } while (next && (win.x <= src.cols));

  // add padding
  if (win.x > padding) {
    win.x -= padding;
  }
  if (win.y > padding) {
    win.y -= padding;
  }
  if ((win.width + win.x + padding) < src.cols) {
    win.width += padding;
  }
  if ((win.height + win.y + padding) < src.rows) {
    win.height += padding;
  }

  return win;
}

template
cv::Rect CropMask(const cv::Mat& src, uchar point, int padding);

cv::Mat colorReduce(const cv::Mat& image, int div) {
  cv::Mat out_img;
  cv::Mat lookUpTable(1, 256, CV_8U);
  uchar* p = lookUpTable.data;
  const int div_2 = div / 2;
  for ( int i = 0; i < 256; ++i ) {
    p[i] = i / div * div + div_2;
  }
  cv::LUT(image, lookUpTable, out_img);
  return out_img;
}

void fillEdgeImage(const cv::Mat& edgesIn, cv::Mat* filledEdgesOut) {
  cv::Mat edgesNeg = edgesIn.clone();
  cv::Scalar val(255, 255, 255);
  cv::floodFill(edgesNeg, cv::Point(0, 0), val);
  cv::floodFill(edgesNeg, cv::Point(edgesIn.cols - 1, edgesIn.rows - 1), val);
  cv::floodFill(edgesNeg, cv::Point(0, edgesIn.rows - 1), val);
  cv::floodFill(edgesNeg, cv::Point(edgesIn.cols - 1, 0), val);
  cv::bitwise_not(edgesNeg, edgesNeg);
  *filledEdgesOut = (edgesNeg | edgesIn);
  return;
}

void CenterObjectAndFillBg(const cv::Mat& in_img, const bool fill_bg,
                           cv::Mat* out_img) {
  cv::Mat mask, crop_mask;
  if (in_img.channels() > 1) {
    cv::Mat in_img_gray;
    cv::cvtColor(in_img, in_img_gray, CV_BGR2GRAY);
    cv::threshold(in_img_gray, mask, 0, 255,
                  CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
  } else {
    cv::threshold(in_img, mask, 0, 255,
                  CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
  }
  cv::Rect crop_rect = CropMask(mask, mask.at<uchar>(0, 0), 2);

  if (fill_bg) {
    cv::Mat temp_img = in_img(crop_rect);
    fillEdgeImage(mask, &mask);
    crop_mask = mask(crop_rect).clone();
    *out_img = cv::Mat::zeros(crop_rect.size(), in_img.type());
    temp_img.copyTo(*out_img, crop_mask);
  } else {
    *out_img = in_img(crop_rect).clone();
  }
}

cv::Mat AspectKeepingResizeAndPad(const cv::Mat& in_img,
                                  const int new_width, const int new_height,
                                  const int pad_type,  const cv::Scalar pad_val,
                                  const int interp_mode) {
  cv::Mat img_resized;
  float orig_aspect = static_cast<float>(in_img.cols) / in_img.rows;
  float new_aspect = static_cast<float>(new_width) / new_height;

  if (orig_aspect > new_aspect) {
    int height = floor(static_cast<float>(new_width) / orig_aspect);
    cv::resize(in_img, img_resized, cv::Size(new_width, height), 0, 0,
               interp_mode);
    cv::Size resSize = img_resized.size();
    int padding = floor((new_height - resSize.height) / 2.0);
    cv::copyMakeBorder(img_resized, img_resized, padding,
                       new_height - resSize.height - padding, 0, 0,
                       pad_type, pad_val);
  } else {
    int width = floor(orig_aspect * new_height);
    cv::resize(in_img, img_resized, cv::Size(width, new_height), 0, 0,
               interp_mode);
    cv::Size resSize = img_resized.size();
    int padding = floor((new_width - resSize.width) / 2.0);
    cv::copyMakeBorder(img_resized, img_resized, 0, 0, padding,
                       new_width - resSize.width - padding,
                       pad_type, pad_val);
  }
  return img_resized;
}

cv::Mat AspectKeepingResizeBySmall(const cv::Mat& in_img,
                                   const int new_width,
                                   const int new_height,
                                   const int interp_mode) {
  cv::Mat img_resized;
  float orig_aspect = static_cast<float>(in_img.cols) / in_img.rows;
  float new_aspect = static_cast<float> (new_width) / new_height;

  if (orig_aspect < new_aspect) {
    int height = floor(static_cast<float>(new_width) / orig_aspect);
    cv::resize(in_img, img_resized, cv::Size(new_width, height), 0, 0,
               interp_mode);
  } else {
    int width = floor(orig_aspect * new_height);
    cv::resize(in_img, img_resized, cv::Size(width, new_height), 0, 0,
               interp_mode);
  }
  return img_resized;
}

cv::Mat AspectKeepingResizeByHeightAndPad(const cv::Mat& in_img,
                                          const int new_width, const int new_height,
                                          const float scale_aspect,
                                          const int pad_type,  const cv::Scalar pad_val,
                                          const int interp_mode) {
  cv::Mat img_resized;
  float orig_aspect = static_cast<float>(in_img.cols) / in_img.rows;
  float new_aspect = static_cast<float>(new_width) / new_height;

  if (scale_aspect * orig_aspect > new_aspect) {
    cv::resize(in_img, img_resized, cv::Size(new_width, new_height), 0, 0,
               interp_mode);
  } else {
    int width = floor(scale_aspect * orig_aspect * new_height);
    cv::resize(in_img, img_resized, cv::Size(width, new_height), 0, 0,
               interp_mode);
    cv::Size resSize = img_resized.size();

    int padding = round((new_width - resSize.width) * 0.5);
    cv::copyMakeBorder(img_resized, img_resized, 0, 0, padding,
                       new_width - resSize.width - padding,
                       pad_type, pad_val);
  }
  return img_resized;
}

cv::Mat AspectKeepingResizeByHeightAndRandomPad(const cv::Mat& in_img,
                                                const int new_width, const int new_height,
                                                const float scale_aspect,
                                                const int pad_type,  const cv::Scalar pad_val,
                                                const int interp_mode) {
  cv::Mat img_resized;
  float orig_aspect = static_cast<float>(in_img.cols) / in_img.rows;
  float new_aspect = static_cast<float>(new_width) / new_height;

  if (scale_aspect * orig_aspect > new_aspect) {
    cv::resize(in_img, img_resized, cv::Size(new_width, new_height), 0, 0,
               interp_mode);
  } else {
    int width = floor(scale_aspect * orig_aspect * new_height);
    cv::resize(in_img, img_resized, cv::Size(width, new_height), 0, 0,
               interp_mode);
    cv::Size resSize = img_resized.size();

    float prob;
    caffe_rng_uniform(1, 0.f, 1.f, &prob);

    int padding = round((new_width - resSize.width) * prob);
    cv::copyMakeBorder(img_resized, img_resized, 0, 0, padding,
                       new_width - resSize.width - padding,
                       pad_type, pad_val);
  }
  return img_resized;
}

void constantNoise(const int n, const vector<uchar>& val, cv::Mat* image) {
  const int cols = image->cols;
  const int rows = image->rows;

  if (image->channels() == 1) {
    for (int k = 0; k < n; ++k) {
      const int i = caffe_rng_rand() % cols;
      const int j = caffe_rng_rand() % rows;
      uchar* ptr = image->ptr<uchar>(j);
      ptr[i]= val[0];
    }
  } else if (image->channels() == 3) {  // color image
    for (int k = 0; k < n; ++k) {
      const int i = caffe_rng_rand() % cols;
      const int j = caffe_rng_rand() % rows;
      cv::Vec3b* ptr = image->ptr<cv::Vec3b>(j);
      (ptr[i])[0] = val[0];
      (ptr[i])[1] = val[1];
      (ptr[i])[2] = val[2];
    }
  }
}

cv::Mat ApplyResize(const cv::Mat& in_img, const ResizeParameter& param) {
  cv::Mat out_img;

  // Reading parameters
  const int new_height = param.height();
  const int new_width = param.width();
  const float scale_aspect = param.scale_aspect();

  int pad_mode = cv::BORDER_CONSTANT;
  switch (param.pad_mode()) {
    case ResizeParameter_Pad_mode_CONSTANT:
      break;
    case ResizeParameter_Pad_mode_MIRRORED:
      pad_mode = cv::BORDER_REFLECT101;
      break;
    case ResizeParameter_Pad_mode_REPEAT_NEAREST:
      pad_mode = cv::BORDER_REPLICATE;
      break;
    case ResizeParameter_Pad_mode_MEAN_VALUE:
      break;
    default:
      LOG(FATAL) << "Unknown pad mode.";
  }

  int interp_mode = cv::INTER_LINEAR;
  int num_interp_mode = param.interp_mode_size();
  if (num_interp_mode > 0) {
    vector<float> probs(num_interp_mode, 1.f / num_interp_mode);
    int prob_num = roll_weighted_die(probs);
    switch (param.interp_mode(prob_num)) {
      case ResizeParameter_Interp_mode_AREA:
        interp_mode = cv::INTER_AREA;
        break;
      case ResizeParameter_Interp_mode_CUBIC:
        interp_mode = cv::INTER_CUBIC;
        break;
      case ResizeParameter_Interp_mode_LINEAR:
        interp_mode = cv::INTER_LINEAR;
        break;
      case ResizeParameter_Interp_mode_NEAREST:
        interp_mode = cv::INTER_NEAREST;
        break;
      case ResizeParameter_Interp_mode_LANCZOS4:
        interp_mode = cv::INTER_LANCZOS4;
        break;
      default:
        LOG(FATAL) << "Unknown interp mode.";
    }
  }

  cv::Scalar pad_val = cv::Scalar(0, 0, 0);
  const int img_channels = in_img.channels();
  if (param.pad_value_size() > 0) {
    CHECK(param.pad_value_size() == 1 ||
          param.pad_value_size() == img_channels) <<
        "Specify either 1 pad_value or as many as channels: " << img_channels;
    vector<float> pad_values;
    for (int i = 0; i < param.pad_value_size(); ++i) {
      pad_values.push_back(param.pad_value(i));
    }
    if (img_channels > 1 && param.pad_value_size() == 1) {
      // Replicate the pad_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        pad_values.push_back(pad_values[0]);
      }
    }
    pad_val = cv::Scalar(pad_values[0], pad_values[1], pad_values[2]);
  }

  if (param.pad_mode() == ResizeParameter_Pad_mode_MEAN_VALUE) {
    std::vector<float> pad_values;
    std::vector<cv::Mat> channels;
    cv::Mat mean, std;
    split(in_img, channels);
    for (int i = 0; i < channels.size(); i++) {
      meanStdDev(channels[i], mean, std);
      pad_values.push_back(mean.at<double>(0, 0));
    }
    pad_val = cv::Scalar(pad_values[0], pad_values[1], pad_values[2]);
  }

  switch (param.resize_mode()) {
    case ResizeParameter_Resize_mode_WARP:
      cv::resize(in_img, out_img, cv::Size(new_width, new_height), 0, 0,
                 interp_mode);
      break;
    case ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD:
      out_img = AspectKeepingResizeAndPad(in_img, new_width, new_height,
                                          pad_mode, pad_val, interp_mode);
      break;
    case ResizeParameter_Resize_mode_FIT_SMALL_SIZE:
      out_img = AspectKeepingResizeBySmall(in_img, new_width, new_height,
                                           interp_mode);
      break;
    case ResizeParameter_Resize_mode_FIT_HEIGHT_AND_PAD:
      out_img = AspectKeepingResizeByHeightAndPad(in_img, new_width, new_height, scale_aspect,
                                                  pad_mode, pad_val, interp_mode);
      break;
    case ResizeParameter_Resize_mode_FIT_HEIGHT_AND_RANDOM_PAD:
      out_img = AspectKeepingResizeByHeightAndRandomPad(in_img, new_width, new_height, scale_aspect,
                                                        pad_mode, pad_val, interp_mode);
      break;
    default:
      LOG(INFO) << "Unknown resize mode.";
  }
  return  out_img;
}

cv::Mat ApplyNoise(const cv::Mat& in_img, const NoiseParameter& param) {
  cv::Mat out_img;
  float prob;

  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (param.decolorize() && prob > param.prob()) {
    cv::Mat grayscale_img;
    cv::cvtColor(in_img, grayscale_img, CV_BGR2GRAY);
    cv::cvtColor(grayscale_img, out_img,  CV_GRAY2BGR);
  } else {
    out_img = in_img;
  }

  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (param.do_blur() && prob > param.prob()) {
    int blur_type = caffe_rng_rand() % 3;
    switch (blur_type) {
      case 0:
        cv::GaussianBlur(out_img, out_img, cv::Size(param.blur_size(), param.blur_size()), 0);
        break;
      case 1:
        cv::blur(out_img, out_img, cv::Size(param.blur_size(), param.blur_size()));
        break;
      case 2:
        cv::medianBlur(out_img, out_img, param.blur_size());
        break;
      default:
        break;
    }
  }

  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (param.do_noise() && prob > param.prob()) {
    int noise_type = caffe_rng_rand() % 2;
    cv::RNG rng(caffe_rng_rand());
    cv::Mat noise_mat;
    switch (noise_type) {
      case 0:
        noise_mat.create(out_img.size(), CV_32FC3);
        rng.fill(noise_mat, cv::RNG::NORMAL, -param.noise_size(), param.noise_size());
        noise_mat.convertTo(noise_mat, CV_8U);
        cv::add(out_img, noise_mat, out_img);
        break;
      case 1:
        noise_mat.create(out_img.size(), CV_32FC3);
        rng.fill(noise_mat, cv::RNG::UNIFORM, -param.noise_size(), param.noise_size());
        noise_mat.convertTo(noise_mat, CV_8U);
        cv::add(out_img, noise_mat, out_img);
        break;
      default:
        break;
    }
  }

  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (param.hist_eq() && prob > param.prob()) {
    if (out_img.channels() > 1) {
      cv::Mat ycrcb_image;
      cv::cvtColor(out_img, ycrcb_image, CV_BGR2YCrCb);
      // Extract the L channel
      vector<cv::Mat> ycrcb_planes(3);
      cv::split(ycrcb_image, ycrcb_planes);
      // now we have the L image in ycrcb_planes[0]
      cv::Mat dst;
      cv::equalizeHist(ycrcb_planes[0], dst);
      ycrcb_planes[0] = dst;
      cv::merge(ycrcb_planes, ycrcb_image);
      // convert back to RGB
      cv::cvtColor(ycrcb_image, out_img, CV_YCrCb2BGR);
    } else {
      cv::Mat temp_img;
      cv::equalizeHist(out_img, temp_img);
      out_img = temp_img;
    }
  }

  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (param.clahe() && prob > param.prob()) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    if (out_img.channels() > 1) {
      cv::Mat ycrcb_image;
      cv::cvtColor(out_img, ycrcb_image, CV_BGR2YCrCb);
      // Extract the L channel
      vector<cv::Mat> ycrcb_planes(3);
      cv::split(ycrcb_image, ycrcb_planes);
      // now we have the L image in ycrcb_planes[0]
      cv::Mat dst;
      clahe->apply(ycrcb_planes[0], dst);
      ycrcb_planes[0] = dst;
      cv::merge(ycrcb_planes, ycrcb_image);
      // convert back to RGB
      cv::cvtColor(ycrcb_image, out_img, CV_YCrCb2BGR);
    } else {
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
      clahe->setClipLimit(4);
      cv::Mat temp_img;
      clahe->apply(out_img, temp_img);
      out_img = temp_img;
    }
  }

  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (param.jpeg() > 0 && prob > param.prob()) {
    vector<uchar> buf;
    vector<int> params;
    params.push_back(CV_IMWRITE_JPEG_QUALITY);
    params.push_back(param.jpeg());
    cv::imencode(".jpg", out_img, buf, params);
    out_img = cv::imdecode(buf, CV_LOAD_IMAGE_COLOR);
  }

  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (param.erode() && prob > param.prob()) {
    cv::Mat element = cv::getStructuringElement(
      2, cv::Size(3, 3), cv::Point(1, 1));
    cv::erode(out_img, out_img, element);
  }

  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (param.dilate() && prob > param.prob()) {
    cv::Mat element = cv::getStructuringElement(
      2, cv::Size(3, 3), cv::Point(1, 1));
    cv::dilate(out_img, out_img, element);
  }

  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (param.posterize() && prob > param.prob()) {
    cv::Mat tmp_img;
    tmp_img = colorReduce(out_img);
    out_img = tmp_img;
  }

  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (param.inverse() && prob > param.prob()) {
    cv::Mat tmp_img;
    cv::bitwise_not(out_img, tmp_img);
    out_img = tmp_img;
  }

  vector<uchar> noise_values;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (param.saltpepper_param().value_size() > 0 && prob > param.prob()) {
    CHECK(param.saltpepper_param().value_size() == 1
          || param.saltpepper_param().value_size() == out_img.channels())
        << "Specify either 1 pad_value or as many as channels: "
        << out_img.channels();

    for (int i = 0; i < param.saltpepper_param().value_size(); i++) {
      noise_values.push_back(uchar(param.saltpepper_param().value(i)));
    }
    if (out_img.channels()  > 1
        && param.saltpepper_param().value_size() == 1) {
      // Replicate the pad_value for simplicity
      for (int c = 1; c < out_img.channels(); ++c) {
        noise_values.push_back(uchar(noise_values[0]));
      }
    }
  }
  if (param.saltpepper() && prob > param.prob()) {
    const int noise_pixels_num =
        floor(param.saltpepper_param().fraction()
              * out_img.cols * out_img.rows);
    constantNoise(noise_pixels_num, noise_values, &out_img);
  }

  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (param.convert_to_hsv() && prob > param.prob()) {
    cv::Mat hsv_image;
    cv::cvtColor(out_img, hsv_image, CV_BGR2HSV);
    out_img = hsv_image;
  }
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (param.convert_to_lab() && prob > param.prob()) {
    cv::Mat lab_image;
    out_img.convertTo(lab_image, CV_32F);
    lab_image *= 1.0 / 255;
    cv::cvtColor(lab_image, out_img, CV_BGR2Lab);
  }
  return  out_img;
}

void RandomBrightness(const cv::Mat& in_img, cv::Mat* out_img,
    const float brightness_prob, const float brightness_delta) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < brightness_prob) {
    CHECK_GE(brightness_delta, 0) << "brightness_delta must be non-negative.";
    float delta;
    caffe_rng_uniform(1, -brightness_delta, brightness_delta, &delta);
    AdjustBrightness(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustBrightness(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img) {
  if (fabs(delta) > 0) {
    in_img.convertTo(*out_img, -1, 1, delta);
  } else {
    *out_img = in_img;
  }
}

void RandomContrast(const cv::Mat& in_img, cv::Mat* out_img,
    const float contrast_prob, const float lower, const float upper) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < contrast_prob) {
    CHECK_GE(upper, lower) << "contrast upper must be >= lower.";
    CHECK_GE(lower, 0) << "contrast lower must be non-negative.";
    float delta;
    caffe_rng_uniform(1, lower, upper, &delta);
    AdjustContrast(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustContrast(const cv::Mat& in_img, const float delta,
                    cv::Mat* out_img) {
  if (fabs(delta - 1.f) > 1e-3) {
    in_img.convertTo(*out_img, -1, delta, 0);
  } else {
    *out_img = in_img;
  }
}

void RandomSaturation(const cv::Mat& in_img, cv::Mat* out_img,
    const float saturation_prob, const float lower, const float upper) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < saturation_prob) {
    CHECK_GE(upper, lower) << "saturation upper must be >= lower.";
    CHECK_GE(lower, 0) << "saturation lower must be non-negative.";
    float delta;
    caffe_rng_uniform(1, lower, upper, &delta);
    AdjustSaturation(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustSaturation(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img) {
  if (fabs(delta - 1.f) != 1e-3) {
    // Convert to HSV colorspae.
    cv::cvtColor(in_img, *out_img, CV_BGR2HSV);

    // Split the image to 3 channels.
    vector<cv::Mat> channels;
    cv::split(*out_img, channels);

    // Adjust the saturation.
    channels[1].convertTo(channels[1], -1, delta, 0);
    cv::merge(channels, *out_img);

    // Back to BGR colorspace.
    cvtColor(*out_img, *out_img, CV_HSV2BGR);
  } else {
    *out_img = in_img;
  }
}

void RandomHue(const cv::Mat& in_img, cv::Mat* out_img,
               const float hue_prob, const float hue_delta) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < hue_prob) {
    CHECK_GE(hue_delta, 0) << "hue_delta must be non-negative.";
    float delta;
    caffe_rng_uniform(1, -hue_delta, hue_delta, &delta);
    AdjustHue(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustHue(const cv::Mat& in_img, const float delta, cv::Mat* out_img) {
  if (fabs(delta) > 0) {
    // Convert to HSV colorspae.
    cv::cvtColor(in_img, *out_img, CV_BGR2HSV);

    // Split the image to 3 channels.
    vector<cv::Mat> channels;
    cv::split(*out_img, channels);

    // Adjust the hue.
    channels[0].convertTo(channels[0], -1, 1, delta);
    cv::merge(channels, *out_img);

    // Back to BGR colorspace.
    cvtColor(*out_img, *out_img, CV_HSV2BGR);
  } else {
    *out_img = in_img;
  }
}

void RandomOrderChannels(const cv::Mat& in_img, cv::Mat* out_img,
                         const float random_order_prob) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < random_order_prob) {
    // Split the image to 3 channels.
    vector<cv::Mat> channels;
    cv::split(*out_img, channels);
    CHECK_EQ(channels.size(), 3);

    // Shuffle the channels.
    std::random_shuffle(channels.begin(), channels.end());
    cv::merge(channels, *out_img);
  } else {
    *out_img = in_img;
  }
}

void RandomDistort(const cv::Mat& in_img, cv::Mat* out_img,
                   float random_distort_prob, const RandomDistortParameter& param)
{
  int width = in_img.cols;
  int height = in_img.rows;

  int w_points = param.w_points();
  int h_points = param.h_points();
  float w_noise = param.w_noise();
  float h_noise = param.h_noise();

  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);

  if (prob > random_distort_prob || w_points < 2 || h_points <2) {
    *out_img = in_img;
    return;
  }

  cv::Mat distort_x(h_points, w_points, CV_32FC1);
  cv::Mat distort_y(h_points, w_points, CV_32FC1);

  cv::Mat map_x(in_img.size(), CV_32FC1);
  cv::Mat map_y(in_img.size(), CV_32FC1);

  for (int j = 0; j < h_points; j++) {
    for (int i = 0; i < w_points; i++) {
      caffe_rng_uniform(1, -w_noise, w_noise, &distort_x.at<float>(j, i));
      caffe_rng_uniform(1, -h_noise, h_noise, &distort_y.at<float>(j, i));
    }
  }

  float x11, x12, x21, x22, y11, y12, y21, y22;
  float a, b;
  int x_min, x_max, y_min, y_max;
  for (int j = 0; j < h_points - 1; j++) {
    for (int i = 0; i < w_points - 1; i++) {
      x_min = i * (width - 1) / (w_points - 1);
      x_max = (i + 1) * (width - 1) / (w_points - 1);
      y_min = j * (height - 1) / (h_points - 1);
      y_max = (j + 1) * (height - 1) / (h_points - 1);

      x11 = x_min + width * distort_x.at<float>(j, i);
      x12 = x_max + width * distort_x.at<float>(j, i + 1);
      x21 = x_min + width * distort_x.at<float>(j + 1, i);
      x22 = x_max + width * distort_x.at<float>(j + 1, i + 1);

      y11 = y_min + height * distort_y.at<float>(j, i);
      y12 = y_min + height * distort_y.at<float>(j, i + 1);
      y21 = y_max + height * distort_y.at<float>(j + 1, i);
      y22 = y_max + height * distort_y.at<float>(j + 1, i + 1);

      for (int y = y_min; y <= y_max; y++) {
        b = (float)(y - y_min) / (float)(y_max - y_min);
        for (int x = x_min; x <= x_max; x++) {
          a = (float)(x - x_min) / (float)(x_max - x_min);
          map_x.at<float>(y, x) = x11 * (1-a) * (1-b) + x12 * a * (1-b) + x21 * (1-a) * b + x22 * a * b;
          map_y.at<float>(y, x) = y11 * (1-a) * (1-b) + y12 * a * (1-b) + y21 * (1-a) * b + y22 * a * b;
        }
      }
    }
  }
  cv::remap(in_img, *out_img, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
}

void RandomPerspective(const cv::Mat& in_img, cv::Mat* out_img,
                       float random_perspective_prob, const RandomPerspectiveParameter& param)
{
  float noise = param.noise();
  float translation = param.translation();
  float perspective = param.perspective();

  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);

  if (prob < random_perspective_prob && noise > 0.0) {
    float map[9];
    caffe_rng_uniform(9, -noise, noise, map);

    map[0] = map[0] + 1.0f;
    map[4] = map[4] + 1.0f;

    map[2] = map[2] * in_img.cols * translation;
    map[5] = map[5] * in_img.rows * translation;

    map[6] = map[6] * perspective;
    map[7] = map[7] * perspective;
    map[8] = map[8] * perspective + 1.0f;

    cv::Mat p_mat(3, 3, CV_32F, map);
    cv::warpPerspective(in_img, *out_img, p_mat, in_img.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
  } else {
    *out_img = in_img;
  }
}

cv::Mat ApplyDistort(const cv::Mat& in_img, const DistortionParameter& param) {
  cv::Mat out_img = in_img;
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);

  if (prob > 0.5) {
    // Do random brightness distortion.
    RandomBrightness(out_img, &out_img, param.brightness_prob(),
                     param.brightness_delta());

    // Do random contrast distortion.
    RandomContrast(out_img, &out_img, param.contrast_prob(),
                   param.contrast_lower(), param.contrast_upper());

    // Do random saturation distortion.
    RandomSaturation(out_img, &out_img, param.saturation_prob(),
                     param.saturation_lower(), param.saturation_upper());

    // Do random hue distortion.
    RandomHue(out_img, &out_img, param.hue_prob(), param.hue_delta());

    // Do random reordering of the channels.
    RandomOrderChannels(out_img, &out_img, param.random_order_prob());
  } else {
    // Do random brightness distortion.
    RandomBrightness(out_img, &out_img, param.brightness_prob(),
                     param.brightness_delta());

    // Do random saturation distortion.
    RandomSaturation(out_img, &out_img, param.saturation_prob(),
                     param.saturation_lower(), param.saturation_upper());

    // Do random hue distortion.
    RandomHue(out_img, &out_img, param.hue_prob(), param.hue_delta());

    // Do random contrast distortion.
    RandomContrast(out_img, &out_img, param.contrast_prob(),
                   param.contrast_lower(), param.contrast_upper());

    // Do random reordering of the channels.
    RandomOrderChannels(out_img, &out_img, param.random_order_prob());
  }

  RandomDistort(out_img, &out_img, param.random_distort_prob(), param.random_distort_param());

  RandomPerspective(out_img, &out_img, param.random_perspective_prob(), param.random_perspective_param());

  return out_img;
}
#endif  // USE_OPENCV

}  // namespace caffe
