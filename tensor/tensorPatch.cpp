#ifndef TENSOR_PATCH_
#define TENSOR_PATCH_

#include "../include/tensor.h"

using std::min;

template <typename DT>
XPU_KERNEL(kernel_relu_fprop) (const int knum, const DT* sdata, DT* ddata) {
    kernel_for (i, knum)
        ddata[i] = sdata[i] >= DT(0) ? sdata[i] : DT(0);
};

template <typename DT>
XPU_KERNEL(kernel_relu_bprop) (const int knum, DT* sdiff, const DT* ddiff) {
    kernel_for (i, knum)
        sdiff[i] = sdiff[i] >= DT(0) ? ddiff[i] : DT(0);
};

template <typename XPU, typename DT>
void Tensor<XPU, DT>::relu_fprop (const Tensor<XPU, DT>& data) {
    const int N = data.size();
    XPU_KERNEL_LAUNCH (kernel_relu_fprop, XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream(),
        N, data.dptr, dptr);
    XPU::check_sync ("kernel_relu_fprop");
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::relu_bprop (const Tensor<XPU, DT>& grad) {
    const int N = grad.size();
    XPU_KERNEL_LAUNCH (kernel_relu_bprop, XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream(),
        N, dptr, grad.dptr);
    XPU::check_sync ("kernel_relu_bprop");
}
#ifdef __CUDACC__
template void TensorGPUf::relu_fprop (const TensorGPUf& data);
template void TensorGPUd::relu_fprop (const TensorGPUd& data);
template void TensorGPUf::relu_bprop (const TensorGPUf& grad);
template void TensorGPUd::relu_bprop (const TensorGPUd& grad);
#else
template void TensorCPUf::relu_fprop (const TensorCPUf& data);
template void TensorCPUd::relu_fprop (const TensorCPUd& data);
template void TensorCPUf::relu_bprop (const TensorCPUf& grad);
template void TensorCPUd::relu_bprop (const TensorCPUd& grad);
#endif



template <typename DT>
XPU_KERNEL(kernel_resize_bilinear_fprop) (const int knum, const DT hratio, const DT wratio,
    const int scols, const int srows, const int schls,
    const int dcols, const int drows, const int dchls, const DT* sdata, DT* ddata) {
    kernel_for (index, knum) {
        const int dx = (index) % dcols;
        const int dy = (index / dcols) % drows;
        const int dc = (index / dcols / drows) % schls;
        const int dn = (index / dcols / dcols / schls);

        const DT sh = dy / hratio;
        const DT sw = dx / wratio;

        const int syu = floor(sh);
        const int sxl = floor(sw);
        const int syd = sh < srows - 1 ? ceil(sh) : srows - 1;
        const int sxr = sw < scols - 1 ? ceil(sw) : scols - 1;

        const int rowu = (dn * schls + dc) * srows + syu;
        const int rowd = (dn * schls + dc) * srows + syd;

        const DT ul = sdata[rowu * scols + sxl];
        const DT ur = sdata[rowu * scols + sxr];
        const DT dl = sdata[rowd * scols + sxl];
        const DT dr = sdata[rowd * scols + sxr];

        const DT yinter = sh - syu;
        const DT xinter = sw - sxl;
        const DT u = ul + (ur - ul) * xinter;
        const DT d = dl + (dr - dl) * xinter;

        ddata[index] = u + (d - u) * yinter;
    }
}

template <typename DT>
XPU_KERNEL(kernel_resize_bilinear_bprop) (const int knum, const DT hratio, const DT wratio,
    const int scols, const int srows, const int schls,
    const int dcols, const int drows, const int dchls, DT* sdiff, const DT* ddiff) {
    kernel_for (index, knum) {
        const int dx = (index) % dcols;
        const int dy = (index / dcols) % drows;
        const int dc = (index / dcols / drows) % schls;
        const int dn = (index / dcols / dcols / schls);

        const DT sh = dy / hratio;
        const DT sw = dx / wratio;

        const int syu = floor(sh);
        const int sxl = floor(sw);
        const int syd = sh < srows - 1 ? ceil(sh) : srows - 1;
        const int sxr = sw < scols - 1 ? ceil(sw) : scols - 1;

        const int rowu = (dn * schls + dc) * srows + syu;
        const int rowd = (dn * schls + dc) * srows + syd;

        const DT yinter = sh - syu;
        const DT xinter = sw - sxl;
        const DT u = (1 - yinter) * ddiff[index];
        const DT d = yinter * ddiff[index];
#ifdef __CUDACC__
        atomicAdd (&sdiff[rowu * scols + sxl], u * (1 - xinter));
        atomicAdd (&sdiff[rowu * scols + sxr], u * xinter);
        atomicAdd (&sdiff[rowd * scols + sxl], d * (1 - xinter));
        atomicAdd (&sdiff[rowd * scols + sxr], d * xinter);
#else
#pragma omp atomic
        sdiff[rowu * scols + sxl] += u * (1 - xinter);
#pragma omp atomic
        sdiff[rowu * scols + sxr] += u * xinter;
#pragma omp atomic
        sdiff[rowd * scols + sxl] += d * (1 - xinter);
#pragma omp atomic
        sdiff[rowd * scols + sxr] += d * xinter;
#endif
    }
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::upsample_fprop (const Tensor<XPU, DT>& data, const int ksize) {
    CHECK_EQ (data.cols(), cols()/ksize);
    CHECK_EQ (data.rows(), rows()/ksize);
    XPU_KERNEL_LAUNCH (kernel_resize_bilinear_fprop, XPU::get_blocks(size()), CUDA_NUM_THREADS, 0, get_calc_stream(),
        size(), DT(ksize), DT(ksize), data.cols(), data.rows(), data.chls(), cols(), rows(), chls(), data.dptr, dptr);
    XPU::check_sync ("kernel_resize_bilinear_fprop");
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::upsample_bprop (const Tensor<XPU, DT>& grad, const int ksize) {
    CHECK_EQ (cols(), grad.cols()/ksize);
    CHECK_EQ (rows(), grad.rows()/ksize);
    mem_set (0);
    XPU_KERNEL_LAUNCH (kernel_resize_bilinear_bprop, XPU::get_blocks(grad.size()), CUDA_NUM_THREADS, 0, get_calc_stream(),
        grad.size(), DT(ksize), DT(ksize), cols(), rows(), chls(), grad.cols(), grad.rows(), grad.chls(), dptr, grad.dptr);
    XPU::check_sync ("kernel_resize_bilinear_bprop");
}
#ifdef __CUDACC__
template void TensorGPUf::upsample_fprop (const TensorGPUf& data, const int ksize);
template void TensorGPUd::upsample_fprop (const TensorGPUd& data, const int ksize);
template void TensorGPUf::upsample_bprop (const TensorGPUf& grad, const int ksize);
template void TensorGPUd::upsample_bprop (const TensorGPUd& grad, const int ksize);
#else
template void TensorCPUf::upsample_fprop (const TensorCPUf& data, const int ksize);
template void TensorCPUd::upsample_fprop (const TensorCPUd& data, const int ksize);
template void TensorCPUf::upsample_bprop (const TensorCPUf& grad, const int ksize);
template void TensorCPUd::upsample_bprop (const TensorCPUd& grad, const int ksize);
#endif



// l 面积大
// s 面积小
template <typename DT>
XPU_KERNEL(kernel_expand_area) (const int knum, const int chlt, const int scols, const int srows, const int schls,
    const int lcols, const int lrows, const int lchls, const DT* sdata, DT* ldata) {
    kernel_for (index, knum) {
        const int lx = index % lcols;
        const int ly = index / lcols % lrows;
        const int lc = index / lcols / lrows % lchls;
        const int ln = index / lcols / lrows / lchls;

        const int sx = lx / chlt;
        const int sy = ly / chlt;
        const int cx = lx % chlt;
        const int cy = ly % chlt;

        const int sc = lc * chlt * chlt + cy * chlt + cx;
        const int sn = ln;

        const int lp = ln * lchls * lrows * lcols + lc * lrows * lcols + ly * lcols + lx;
        const int sp = sn * schls * srows * scols + sc * srows * scols + sy * scols + sx;

        ldata[lp] = sdata[sp];
    }
}

// l 面积大
// s 面积小
template <typename DT>
XPU_KERNEL(kernel_expand_chls) (const int knum, const int chlt, const int scols, const int srows, const int schls,
    const int lcols, const int lrows, const int lchls, DT* sdata, const DT* ldata) {
    kernel_for (index, knum) {
        const int lx = index % lcols;
        const int ly = index / lcols % lrows;
        const int lc = index / lcols / lrows % lchls;
        const int ln = index / lcols / lrows / lchls;

        const int sx = lx / chlt;
        const int sy = ly / chlt;
        const int cx = lx % chlt;
        const int cy = ly % chlt;

        const int sc = lc * chlt * chlt + cy * chlt + cx;
        const int sn = ln;

        const int lp = ln * lchls * lrows * lcols + lc * lrows * lcols + ly * lcols + lx;
        const int sp = sn * schls * srows * scols + sc * srows * scols + sy * scols + sx;

        sdata[sp] = ldata[lp];
    }
}

template <typename DT>
XPU_KERNEL(kernel_reduce_mask) (const int knum, const int cols, const int rows, const int chls, const DT thresh,
    DT* sdata, const DT* ldata) {
    kernel_for (index, knum) {
        const int sx = index % cols;
        const int sy = index / cols % rows;
        const int sn = index / cols / rows;

        const int lp = sn * chls * rows * cols + sy * cols + sx;
        const int sp = sn * rows * cols + sy * cols + sx;

        DT maxv = 0, maxi = 0;
        for (int lc = 0; lc < chls; ++lc) {
            const int tp = lp + lc * rows * cols;
            if (ldata[tp] > maxv) { maxv = ldata[tp];  maxi = lc; }
        }
        sdata[sp] = maxv >= thresh ? maxi : DT(-1);
    }
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::expand_area (const Tensor<XPU, DT>& in, const int chlt) {
    CHECK_EQ (in.cols(), cols()/chlt);
    CHECK_EQ (in.rows(), rows()/chlt);
    XPU_KERNEL_LAUNCH (kernel_expand_area, XPU::get_blocks(in.size()), CUDA_NUM_THREADS, 0, get_calc_stream(),
        in.size(), chlt, in.cols(), in.rows(), in.chls(), cols(), rows(), chls(), in.dptr, dptr);
    XPU::check_sync ("kernel_expand_area");
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::expand_chls (const Tensor<XPU, DT>& in, const int chlt) {
    CHECK_EQ (cols(), in.cols()/chlt);
    CHECK_EQ (rows(), in.rows()/chlt);
    XPU_KERNEL_LAUNCH (kernel_expand_chls, XPU::get_blocks(in.size()), CUDA_NUM_THREADS, 0, get_calc_stream(),
        in.size(), chlt, cols(), rows(), chls(), in.cols(), in.rows(), in.chls(), dptr, in.dptr);
    XPU::check_sync ("kernel_expand_chls");
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::reduce_mask (const Tensor<XPU, DT>& in, const DT thresh) {
    CHECK_EQ (cols(), in.cols());
    CHECK_EQ (rows(), in.rows());
    CHECK_EQ (chls(), 1);
    XPU_KERNEL_LAUNCH (kernel_reduce_mask, XPU::get_blocks(size()), CUDA_NUM_THREADS, 0, get_calc_stream(),
        size(), in.cols(), in.rows(), in.chls(), thresh, dptr, in.dptr);
    XPU::check_sync ("kernel_reduce_mask");
}
#ifdef __CUDACC__
template void TensorGPUf::expand_area (const TensorGPUf& in, const int chlt);
template void TensorGPUd::expand_area (const TensorGPUd& in, const int chlt);
template void TensorGPUf::expand_chls (const TensorGPUf& in, const int chlt);
template void TensorGPUd::expand_chls (const TensorGPUd& in, const int chlt);
template void TensorGPUf::reduce_mask (const TensorGPUf& in, const float thresh);
template void TensorGPUd::reduce_mask (const TensorGPUd& in, const double thresh);
#else
template void TensorCPUf::expand_area (const TensorCPUf& in, const int chlt);
template void TensorCPUd::expand_area (const TensorCPUd& in, const int chlt);
template void TensorCPUf::expand_chls (const TensorCPUf& in, const int chlt);
template void TensorCPUd::expand_chls (const TensorCPUd& in, const int chlt);
template void TensorCPUf::reduce_mask (const TensorCPUf& in, const float thresh);
template void TensorCPUd::reduce_mask (const TensorCPUd& in, const double thresh);
#endif



XPU_KERNEL(im2col) (
  const int knum, const float* data_im, float* data_col, const int rows, const int cols, const int chls,
  const int krow, const int kcol, const int pad, const int stride, const int h_col, const int w_col)
{ const int p_col = h_col * w_col;
  kernel_for (index, knum)
  { int w_out = index % w_col;
    int h_out = index / w_col % h_col;
    int c_in  = index / w_col / h_col;
    int sw  = w_out * stride - pad;
    int sh  = h_out * stride - pad;
    float* ptr_col = data_col + c_in * (krow * kcol * p_col) + h_out * w_col + w_out;
    const float* ptr_im = data_im  + c_in * (rows * cols) + sh * cols + sw;
    for (int i = 0; i < krow; ++i)
      for (int j = 0; j < kcol; ++j)
      { int h = sh + i;
        int w = sw + j;
        ptr_col[(i*kcol+j) * p_col] = (h >= 0 && w >= 0 && h < rows && w < cols) ? ptr_im[i * cols + j] : 0;
      }
  }
}

XPU_KERNEL(col2im) (
  const int knum, float* data_im, const float* data_col, const int rows, const int cols, const int chls,
  const int krow, const int kcol, const int pad, const int stride, const int h_col, const int w_col)
{ const int p_col = h_col * w_col;
  kernel_for (index, knum)
  { float val = 0;
    int w = index % cols + pad;
    int h = (index / cols) % rows + pad;
    int c = index / (cols * rows);
    // compute the start and end of the output
    int w_col_start = (w < kcol) ? 0 : (w - kcol) / stride + 1;
    int h_col_start = (h < krow) ? 0 : (h - krow) / stride + 1;
    int w_col_end = min(w / stride + 1, w_col);
    int h_col_end = min(h / stride + 1, h_col);
    int offset = (c * krow * kcol + h * kcol + w) * p_col;
    int coeff_h_col = (1 - stride * kcol * h_col) * w_col;
    int coeff_w_col = (1 - stride * p_col);
    for (int i = h_col_start; i < h_col_end; ++i)
      for (int j = w_col_start; j < w_col_end; ++j)
        val += data_col[offset + i * coeff_h_col + j * coeff_w_col];
    data_im[index] = val;
  }
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::im2col_fprop (const Kernal& p, Tensor<XPU, DT>& im_col)
{ const int N = chls() * p.h_col * p.w_col;
  XPU_KERNEL_LAUNCH (im2col, XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream(),
    N, dptr, im_col.dptr, rows(), cols(), chls(), p.krow, p.kcol, p.pad, p.stride, p.h_col, p.w_col);
};

template <typename XPU, typename DT>
void Tensor<XPU, DT>::col2im_bprop (const Kernal& p, Tensor<XPU, DT>& im_col)
{ const int N = chls() * p.h_col * p.w_col;
  XPU_KERNEL_LAUNCH (col2im, XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream(),
    N, dptr, im_col.dptr, rows(), cols(), chls(), p.krow, p.kcol, p.pad, p.stride, p.h_col, p.w_col);
}

#ifdef __CUDACC__
template void TensorGPUf::im2col_fprop (const Kernal& k, TensorGPUf& im_col);
template void TensorGPUf::col2im_bprop (const Kernal& k, TensorGPUf& im_col);
#else
template void TensorCPUf::col2im_bprop (const Kernal& k, TensorCPUf& im_col);
template void TensorCPUf::im2col_fprop (const Kernal& k, TensorCPUf& im_col);
#endif

#endif
