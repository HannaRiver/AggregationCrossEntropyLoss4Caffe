#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"

namespace caffe {
  template <typename Dtype>
  void BNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());
    if (top.size() > 1) {
        // top blob for batch mean
        top[1]->Reshape(1, C_, 1, 1);
    }
    if (top.size() > 2) {
        // top blob for batch variance
        top[2]->Reshape(1, C_, 1, 1);
    }

    x_norm_.Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());

    // mean
    spatial_mean_.Reshape(N_, C_, 1, 1);
    batch_mean_.Reshape(1, C_, 1, 1);
    // variance
    spatial_variance_.Reshape(N_, C_, 1, 1);
    batch_variance_.Reshape(1, C_, 1, 1);
    // buffer blob
    buffer_blob_.Reshape(N_, C_, H_, W_);

    // fill spatial multiplier
    spatial_sum_multiplier_.Reshape(1, 1, H_, W_);
    Dtype* spatial_multipl_data = spatial_sum_multiplier_.mutable_cpu_data();
    caffe_set(spatial_sum_multiplier_.count(), Dtype(1),
        spatial_multipl_data);
    caffe_set(spatial_sum_multiplier_.count(), Dtype(0),
        spatial_sum_multiplier_.mutable_cpu_diff());
    // fill batch multiplier
    batch_sum_multiplier_.Reshape(N_, 1, 1, 1);
    Dtype* batch_multiplier_data = batch_sum_multiplier_.mutable_cpu_data();
    caffe_set(batch_sum_multiplier_.count(), Dtype(1),
        batch_multiplier_data);
    caffe_set(batch_sum_multiplier_.count(), Dtype(0),
        batch_sum_multiplier_.mutable_cpu_diff());
  }
  template <typename Dtype>
  void BNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // Figure out the dimensions
    N_ = bottom[0]->num();
    C_ = bottom[0]->channels();
    H_ = bottom[0]->height();
    W_ = bottom[0]->width();
    var_eps_ = 1e-9;

    // Check if we need to set up the weights
    if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping parameter initialization";
    } else {
      this->blobs_.resize(2);

      // fill scale with scale_filler
      this->blobs_[0].reset(new Blob<Dtype>(1, C_, 1, 1));
      shared_ptr<Filler<Dtype> > scale_filler(GetFiller<Dtype>(
          this->layer_param_.bn_param().scale_filler()));
      scale_filler->Fill(this->blobs_[0].get());

      // fill shift with shift_filler
      this->blobs_[1].reset(new Blob<Dtype>(1, C_, 1, 1));
      shared_ptr<Filler<Dtype> > shift_filler(GetFiller<Dtype>(
          this->layer_param_.bn_param().shift_filler()));
      shift_filler->Fill(this->blobs_[1].get());
    }  // parameter initialization
    this->param_propagate_down_.resize(this->blobs_.size(), true);
  }

  template <typename Dtype>
  void BNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* const_top_data = top[0]->cpu_data();

    const Dtype* scale_data = this->blobs_[0]->cpu_data();
    const Dtype* shift_data = this->blobs_[1]->cpu_data();

    switch (this->layer_param_.bn_param().bn_mode()) {
    case BNParameter_BNMode_LEARN:
      // put the squares of bottom into buffer_blob_
      caffe_powx(bottom[0]->count(), bottom_data, Dtype(2),
          buffer_blob_.mutable_cpu_data());

      // computes variance using var(X) = E(X^2) - (EX)^2
      // EX across spatial
      caffe_cpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_,
          Dtype(1. / (H_ * W_)), bottom_data,
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          spatial_mean_.mutable_cpu_data());
      // EX across batch
      caffe_cpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1. / N_),
          spatial_mean_.cpu_data(),
          batch_sum_multiplier_.cpu_data(), Dtype(0),
          batch_mean_.mutable_cpu_data());

      // E(X^2) across spatial
      caffe_cpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_,
          Dtype(1. / (H_ * W_)), buffer_blob_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          spatial_variance_.mutable_cpu_data());
      // E(X^2) across batch
      caffe_cpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1. / N_),
          spatial_variance_.cpu_data(),
          batch_sum_multiplier_.cpu_data(), Dtype(0),
          batch_variance_.mutable_cpu_data());

      caffe_powx(batch_mean_.count(), batch_mean_.cpu_data(), Dtype(2),
          buffer_blob_.mutable_cpu_data());  // (EX)^2
      caffe_sub(batch_mean_.count(), batch_variance_.cpu_data(),
          buffer_blob_.cpu_data(),
          batch_variance_.mutable_cpu_data());  // variance

      // save top[1] (batch_mean) and top[2] (batch_variance)
      if (top.size() > 1) {
          caffe_copy(batch_mean_.count(), batch_mean_.cpu_data(),
              top[1]->mutable_cpu_data());
      }
      if (top.size() > 2) {
          caffe_copy(batch_variance_.count(), batch_variance_.cpu_data(),
              top[2]->mutable_cpu_data());
      }

      // do mean and variance normalization
      // subtract mean
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_,
          C_, 1, Dtype(1),
          batch_sum_multiplier_.cpu_data(),
          batch_mean_.cpu_data(), Dtype(0),
          spatial_mean_.mutable_cpu_data());

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_,
          H_ * W_, 1, Dtype(-1),
          spatial_mean_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          buffer_blob_.mutable_cpu_data());

      caffe_add(buffer_blob_.count(), bottom_data,
          buffer_blob_.cpu_data(), top_data);

      // normalize variance
      caffe_add_scalar(batch_variance_.count(), var_eps_,
        batch_variance_.mutable_cpu_data());
      caffe_powx(batch_variance_.count(),
          batch_variance_.cpu_data(), Dtype(0.5),
          batch_variance_.mutable_cpu_data());

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_,
          C_, 1, Dtype(1),
          batch_sum_multiplier_.cpu_data(),
          batch_variance_.cpu_data(), Dtype(0),
          spatial_variance_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          N_ * C_, H_ * W_, 1, Dtype(1),
          spatial_variance_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          buffer_blob_.mutable_cpu_data());

      caffe_div(buffer_blob_.count(), const_top_data,
          buffer_blob_.cpu_data(), top_data);

      // Saving x_norm
      caffe_copy(buffer_blob_.count(), const_top_data,
          x_norm_.mutable_cpu_data());
      // scale
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.cpu_data(), scale_data, Dtype(0),
          spatial_variance_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_,
          H_ * W_, 1, Dtype(1),
          spatial_variance_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          buffer_blob_.mutable_cpu_data());
      caffe_mul(buffer_blob_.count(), top_data,
          buffer_blob_.cpu_data(), top_data);

      // shift
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.cpu_data(), shift_data, Dtype(0),
          spatial_mean_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          N_ * C_, H_ * W_, 1, Dtype(1),
          spatial_mean_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          buffer_blob_.mutable_cpu_data());
      caffe_add(buffer_blob_.count(), const_top_data,
          buffer_blob_.cpu_data(), top_data);
      break;
    case BNParameter_BNMode_INFERENCE:
      // scale
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.cpu_data(), scale_data, Dtype(0),
          spatial_variance_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_,
          H_ * W_, 1, Dtype(1),
          spatial_variance_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          buffer_blob_.mutable_cpu_data());
      caffe_mul(buffer_blob_.count(), bottom_data,
          buffer_blob_.cpu_data(), top_data);

      // shift
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.cpu_data(), shift_data, Dtype(0),
          spatial_mean_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          N_ * C_, H_ * W_, 1, Dtype(1),
          spatial_mean_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          buffer_blob_.mutable_cpu_data());
      caffe_add(buffer_blob_.count(), const_top_data,
          buffer_blob_.cpu_data(), top_data);
      break;
    default:
      LOG(FATAL) << "Unknown BN mode.";
    } 
  }

  template <typename Dtype>
  void BNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* shift_diff = this->blobs_[1]->mutable_cpu_diff();
    const Dtype* scale_data = this->blobs_[0]->cpu_data();

    switch (this->layer_param_.bn_param().bn_mode()) {
    case BNParameter_BNMode_LEARN:
      // Propagate layer to parameters
      // gradient w.r.t. scale
      caffe_mul(buffer_blob_.count(), x_norm_.cpu_data(),
          top_diff, buffer_blob_.mutable_cpu_data());
      // EX across spatial
      caffe_cpu_gemv<Dtype>(CblasNoTrans, N_ * C_,
          H_ * W_, Dtype(1), buffer_blob_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          spatial_variance_.mutable_cpu_diff());
      // EX across batch
      caffe_cpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1),
          spatial_variance_.cpu_diff(),
          batch_sum_multiplier_.cpu_data(), Dtype(0), scale_diff);

      // gradient w.r.t. shift
      // EX across spatial
      caffe_cpu_gemv<Dtype>(CblasNoTrans, N_ * C_,
          H_ * W_, Dtype(1), top_diff,
          spatial_sum_multiplier_.cpu_data(),
          Dtype(0), spatial_mean_.mutable_cpu_diff());
      // EX across batch
      caffe_cpu_gemv<Dtype>(CblasTrans, N_, C_,
          Dtype(1), spatial_mean_.cpu_diff(),
          batch_sum_multiplier_.cpu_data(),
          Dtype(0), shift_diff);

      // Propagate down

      // put scale * top_diff to buffer_blob_
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.cpu_data(), scale_data, Dtype(0),
          spatial_variance_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_,
          H_ * W_, 1, Dtype(1),
          spatial_variance_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          buffer_blob_.mutable_cpu_data());
      caffe_mul(buffer_blob_.count(), top_diff, buffer_blob_.cpu_data(),
          buffer_blob_.mutable_cpu_data());

      // use new top diff for computation
      caffe_mul(buffer_blob_.count(),  x_norm_.cpu_data(),
          buffer_blob_.cpu_data(), bottom_diff);
      // EX across spatial
      caffe_cpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_,
          Dtype(1), bottom_diff,
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          spatial_mean_.mutable_cpu_data());
      // EX across batch
      caffe_cpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1),
          spatial_mean_.cpu_data(),
          batch_sum_multiplier_.cpu_data(), Dtype(0),
          batch_mean_.mutable_cpu_data());

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.cpu_data(),
          batch_mean_.cpu_data(), Dtype(0),
          spatial_mean_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_,
          H_ * W_, 1, Dtype(1),
          spatial_mean_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          bottom_diff);

      caffe_mul(buffer_blob_.count(),
          x_norm_.cpu_data(), bottom_diff, bottom_diff);

      // EX across spatial
      caffe_cpu_gemv<Dtype>(CblasNoTrans, N_ * C_,
          H_ * W_, Dtype(1), buffer_blob_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          spatial_mean_.mutable_cpu_data());
      // EX across batch
      caffe_cpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1),
          spatial_mean_.cpu_data(),
          batch_sum_multiplier_.cpu_data(), Dtype(0),
          batch_mean_.mutable_cpu_data());

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.cpu_data(),
          batch_mean_.cpu_data(), Dtype(0),
          spatial_mean_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          N_ * C_, H_ * W_, 1, Dtype(1),
          spatial_mean_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), Dtype(1), bottom_diff);

      caffe_cpu_axpby(buffer_blob_.count(), Dtype(1),
          buffer_blob_.cpu_data(), Dtype(-1. / (N_ * H_ * W_)),
          bottom_diff);

      // put the squares of bottom into buffer_blob_
      caffe_powx(buffer_blob_.count(), bottom_data, Dtype(2),
          buffer_blob_.mutable_cpu_data());

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.cpu_data(),
          batch_variance_.cpu_data(), Dtype(0),
          spatial_variance_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          N_ * C_, H_ * W_, 1, Dtype(1),
          spatial_variance_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          buffer_blob_.mutable_cpu_data());

      caffe_div(buffer_blob_.count(), bottom_diff,
          buffer_blob_.cpu_data(), bottom_diff);
      break;
    case BNParameter_BNMode_INFERENCE:
      // Propagate layer to parameters
      // gradient w.r.t. scale
      caffe_mul(buffer_blob_.count(), bottom_data,
          top_diff, buffer_blob_.mutable_cpu_data());
      // EX across spatial
      caffe_cpu_gemv<Dtype>(CblasNoTrans, N_ * C_,
          H_ * W_, Dtype(1), buffer_blob_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          spatial_variance_.mutable_cpu_diff());
      // EX across batch
      caffe_cpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1),
          spatial_variance_.cpu_diff(),
          batch_sum_multiplier_.cpu_data(), Dtype(0), scale_diff);

      // gradient w.r.t. shift
      // EX across spatial
      caffe_cpu_gemv<Dtype>(CblasNoTrans, N_ * C_,
          H_ * W_, Dtype(1), top_diff,
          spatial_sum_multiplier_.cpu_data(),
          Dtype(0), spatial_mean_.mutable_cpu_diff());
      // EX across batch
      caffe_cpu_gemv<Dtype>(CblasTrans, N_, C_,
          Dtype(1), spatial_mean_.cpu_diff(),
          batch_sum_multiplier_.cpu_data(),
          Dtype(0), shift_diff);

      // Propagate down
      // put scale * top_diff to buffer_blob_
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.cpu_data(), scale_data, Dtype(0),
          spatial_variance_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_,
          H_ * W_, 1, Dtype(1),
          spatial_variance_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), Dtype(0),
          buffer_blob_.mutable_cpu_data());
      caffe_mul(buffer_blob_.count(), top_diff, buffer_blob_.cpu_data(),
          bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown BN mode.";
    }
  }
#ifdef CPU_ONLY
STUB_GPU(BNLayer);
#endif

  INSTANTIATE_CLASS(BNLayer);
  REGISTER_LAYER_CLASS(BN);

template <typename Dtype>
void BatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  moving_average_fraction_ = param.moving_average_fraction();
  use_global_stats_ = this->phase_ == TEST;
  if (param.has_use_global_stats())
    use_global_stats_ = param.use_global_stats();
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);
  eps_ = param.eps();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob<Dtype>(sz));
    this->blobs_[1].reset(new Blob<Dtype>(sz));
    sz[0] = 1;
    this->blobs_[2].reset(new Blob<Dtype>(sz));
    for (int i = 0; i < 3; ++i) {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }
  }
  // Mask statistics from optimization by setting local learning rates
  // for mean, variance, and the bias correction to zero.
  for (int i = 0; i < this->blobs_.size(); ++i) {
    if (this->layer_param_.param_size() == i) {
      ParamSpec* fixed_param_spec = this->layer_param_.add_param();
      fixed_param_spec->set_lr_mult(0.f);
    } else {
      CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
          << "Cannot configure batch normalization statistics as layer "
          << "parameters.";
    }
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() >= 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> sz;
  sz.push_back(channels_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  temp_.ReshapeLike(*bottom[0]);
  x_norm_.ReshapeLike(*bottom[0]);
  sz[0] = bottom[0]->shape(0);
  batch_sum_multiplier_.Reshape(sz);

  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));
  if (spatial_sum_multiplier_.num_axes() == 0 ||
      spatial_sum_multiplier_.shape(0) != spatial_dim) {
    sz[0] = spatial_dim;
    spatial_sum_multiplier_.Reshape(sz);
    Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
    caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
  }

  int numbychans = channels_*bottom[0]->shape(0);
  if (num_by_chans_.num_axes() == 0 ||
      num_by_chans_.shape(0) != numbychans) {
    sz[0] = numbychans;
    num_by_chans_.Reshape(sz);
    caffe_set(batch_sum_multiplier_.count(), Dtype(1),
        batch_sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }

  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
        0 : 1 / this->blobs_[2]->cpu_data()[0];
    caffe_cpu_scale(variance_.count(), scale_factor,
        this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
    caffe_cpu_scale(variance_.count(), scale_factor,
        this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
  } else {
    // compute mean
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), bottom_data,
        spatial_sum_multiplier_.cpu_data(), 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
        mean_.mutable_cpu_data());
  }

  // subtract mean
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, -1, num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 1., top_data);

  if (!use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_sqr<Dtype>(top[0]->count(), top_data,
                     temp_.mutable_cpu_data());  // (X-EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), temp_.cpu_data(),
        spatial_sum_multiplier_.cpu_data(), 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E((X_EX)^2)

    // compute and save moving average
    this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
    this->blobs_[2]->mutable_cpu_data()[0] += 1;
    caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
        moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
    int m = bottom[0]->count()/channels_;
    Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
    caffe_cpu_axpby(variance_.count(), bias_correction_factor,
        variance_.cpu_data(), moving_average_fraction_,
        this->blobs_[1]->mutable_cpu_data());
  }

  // normalize variance
  caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
  caffe_sqrt(variance_.count(), variance_.cpu_data(),
             variance_.mutable_cpu_data());

  // replicate variance to input size
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), variance_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
  caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
  // TODO(cdoersch): The caching is only needed because later in-place layers
  //                 might clobber the data.  Can we skip this if they won't?
  caffe_copy(x_norm_.count(), top_data,
      x_norm_.mutable_cpu_data());
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = top[0]->cpu_diff();
  } else {
    caffe_copy(x_norm_.count(), top[0]->cpu_diff(), x_norm_.mutable_cpu_diff());
    top_diff = x_norm_.cpu_diff();
  }
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (use_global_stats_) {
    caffe_div(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
    return;
  }
  const Dtype* top_data = x_norm_.cpu_data();
  int num = bottom[0]->shape()[0];
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());

  // reshape (broadcast) the above
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff,
      Dtype(-1. / (num * spatial_dim)), bottom_diff);

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(BatchNormLayer);
#endif

INSTANTIATE_CLASS(BatchNormLayer);
REGISTER_LAYER_CLASS(BatchNorm);
}  // namespace caffe
