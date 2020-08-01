#ifndef TENSOR_SHAPE_
#define TENSOR_SHAPE_

#include "../include/tensor.h"

#ifndef __CUDACC__
Shape::Shape () : rows(0), cols(0), chls(0), nums(0) {
    set_dims ();
}

Shape::Shape (const int a, const int b, const int c, const int d) : rows(a), cols(b), chls(c), nums(d) {
    set_dims ();
}

Shape::Shape (const int a, const int b, const int c, const int d, const int e) : rows(a), cols(b), chls(c), nums(d) {
    set_dims ();
    size = e;
    CHECK_LT (size, 2147483648);
}

bool Shape::operator == (const Shape& s) const {
    return (rows == s.rows && cols == s.cols && chls == s.chls && nums == s.nums);
}

bool Shape::operator != (const Shape& s) const {
    return !(*this == s);
}

Shape Shape::section (const int begin, const int end) const {
    const int slices = end + 1 - begin;  // 127 + 1 - 0
    CHECK_GE (slices, 1);
    if      (dims == 4) { CHECK_LE (slices, nums); return Shape (rows, cols, chls, slices); }
    else if (dims == 3) { CHECK_LE (slices, chls); return Shape (rows, cols, slices, nums); }
    else                { CHECK_LE (slices, rows); return Shape (slices, cols, chls, nums); }
}

Shape Shape::upsample (const int ksize) const {
    return Shape (rows*ksize, cols*ksize, chls, nums);
}

Shape Shape::expand_aspp (const int chlp) const {
    return Shape (rows, cols, chls*chlp, nums);
}

Shape Shape::expand_area (const int chlt) const {
    CHECK_EQ (chls % (chlt*chlt), 0);
    return Shape (rows*chlt, cols*chlt, chls/chlt/chlt, nums);
}

Shape Shape::expand_chls (const int chlt) const {
    CHECK_EQ (rows % chlt, 0);
    CHECK_EQ (cols % chlt, 0);
    return Shape (rows/chlt, cols/chlt, chls*chlt*chlt, nums);
}

Shape Shape::reduce_mask () const {
    return Shape (rows, cols, 1, nums);
}

void Shape::set_dims () {
    size = rows * cols * chls * nums;
    CHECK_LT (size, 2147483648);
    if      (nums > 1) dims = 4;
    else if (chls > 1) dims = 3;
    else if (cols > 1) dims = 2;
    else               dims = 1;
}

void Shape::set_chls (const int c) {
    CHECK_EQ ((rows * chls) % c, 0);
    rows = rows * chls / c;
    chls = c;
    set_dims ();
}

void Shape::set_cols (const int c) {
    CHECK_EQ ((rows * cols) % c, 0);
    rows = rows * cols / c;
    cols = c;
    set_dims ();
}

void Shape::print () {
    char shapestr[64]; sprintf (shapestr, "\tshape\t%d\t%d\t%d\t%d\n", cols, rows, chls, nums);
    LOG (INFO) << shapestr;
}

void Shape::print () const {
    char shapestr[64]; sprintf (shapestr, "\tshape\t%d\t%d\t%d\t%d\n", cols, rows, chls, nums);
    LOG (INFO) << shapestr;
}



Shape Kernal::get_pack_size (const Shape& in) {
    CHECK_GE (krow * kcol, 1);
    h_col = (in.rows - 1) / stride + 1;  // TODO
    w_col = (in.cols - 1) / stride + 1;  // TODO
    return Shape (krow * kcol * in.chls, h_col * w_col, 1, in.nums);
}

Shape Kernal::get_pool_size (const Shape& in) {
    CHECK_GE (krow * kcol, 1);
    h_pool = (in.rows - 1) / stride + 1;  // TODO
    w_pool = (in.cols - 1) / stride + 1;  // TODO
    return Shape (h_pool, w_pool, in.chls, in.nums);
}



// 头部规约
void config_head_sizeX_strdX (const Shape& lshape, const Shape& sshape, int& sizeX, int& strdX) {
    sizeX = 1;
    if (sshape.cols == 1)  sizeX *= lshape.cols;  else  return;
    if (sshape.rows == 1)  sizeX *= lshape.rows;  else  return;
    if (sshape.chls == 1)  sizeX *= lshape.chls;  else  return;
    if (sshape.nums == 1)  sizeX *= lshape.nums;  else  return;
}

// 尾部规约 中部规约
void config_tail_sizeX_strdX (const Shape& lshape, const Shape& sshape, int& sizeX, int& strdX) {
    sizeX = 1;
    strdX = 1;
    strdX *= lshape.cols;  if (sshape.cols == lshape.cols)  sizeX *= lshape.cols;  else if (sshape.rows == lshape.rows)  return;
    strdX *= lshape.rows;  if (sshape.rows == lshape.rows)  sizeX *= lshape.rows;  else if (sshape.chls == lshape.chls)  return;
    strdX *= lshape.chls;  if (sshape.chls == lshape.chls)  sizeX *= lshape.chls;  else if (sshape.nums == lshape.nums)  return;
    strdX *= lshape.nums;  if (sshape.nums == lshape.nums)  sizeX *= lshape.nums;  else return;
}

// 两头规约
void config_side_sizeX_strdX (const Shape& lshape, const Shape& sshape, Shape& tshape) {
    if (sshape.rows == lshape.rows)
        tshape = Shape (lshape.rows, sshape.cols, lshape.chls, lshape.nums);
    else
        tshape = Shape (sshape.rows, sshape.cols, lshape.chls, lshape.nums);
}



static int ceil_pow2 (const float x) {
    const int idx = std::ceil(std::log2(x));
    return 1 << idx;
}

static int floor_pow2 (const float x) {
    const int idx = std::floor(std::log2(x));
    return 1 << idx;
}

static bool reduce_head (const Shape& lshape, const Shape& sshape) {
    if (sshape.cols < lshape.cols)  return true;  else if (sshape.cols > 1 && sshape.cols == lshape.cols)  return false;
    if (sshape.rows < lshape.rows)  return true;  else if (sshape.rows > 1 && sshape.rows == lshape.rows)  return false;
    if (sshape.chls < lshape.chls)  return true;  else if (sshape.chls > 1 && sshape.chls == lshape.chls)  return false;
    if (sshape.nums < lshape.nums)  return true;  else return false;
}

static bool reduce_tail (const Shape& lshape, const Shape& sshape) {
    if (sshape.nums < lshape.nums)  return true;  else if (sshape.nums > 1 && sshape.nums == lshape.nums)  return false;
    if (sshape.chls < lshape.chls)  return true;  else if (sshape.chls > 1 && sshape.chls == lshape.chls)  return false;
    if (sshape.rows < lshape.rows)  return true;  else if (sshape.rows > 1 && sshape.rows == lshape.rows)  return false;
    if (sshape.cols < lshape.cols)  return true;  else return false;
}

// TODO
int get_reduce_type (const Shape& lshape, const Shape& sshape) {
    CHECK (sshape.cols == 1 || sshape.cols == lshape.cols);
    CHECK (sshape.rows == 1 || sshape.rows == lshape.rows);
    CHECK (sshape.chls == 1 || sshape.chls == lshape.chls);
    CHECK (sshape.nums == 1 || sshape.nums == lshape.nums);

    const bool rhead = reduce_head (lshape, sshape);
    const bool rtail = reduce_tail (lshape, sshape);

    if (rhead && !rtail)
        return 1;  // reduce head
    else if (!rhead && rtail)
        return 2;  // reduce tail
    else if (rhead && rtail)
        return 3;  // reduce side
    else
        return 4;  // reduce midd
}

// head_reduce : sizeT = sizeX
// tail_reduce : sizeT = strdX / sizeX
int get_reduce_plan (const Shape& lshape, const Shape& sshape, const int sizeT, const int sizeX, unsigned int& blocks, unsigned int& threads) {
    const int sizeData = lshape.size;
    if (2 == get_reduce_type (lshape, sshape) && sizeX >= 1024*8) {
        threads = 1024;
        blocks = (sizeData / sizeT + threads - 1) / threads;
        return 2;  // 单线程单独规约
    }
    else {
        threads = std::min (1024, ceil_pow2 (std::sqrt(sizeT)));
        blocks = sizeData / sizeT;
        return 1;  // 多线程协作规约
    }
}

// head_repeat : sizeT = sizeX
// tail_repeat : sizeT = strdX / sizeX
int get_repeat_plan (const Shape& lshape, const Shape& sshape, const int sizeT, const int sizeX, unsigned int& blocks, unsigned int& threads) {
    const int sizeData = lshape.size;
    if (2 == get_reduce_type (lshape, sshape) && sizeX >= 1024*8) {
        threads = 1024;
        blocks = (sizeData / sizeT + threads - 1) / threads;
        return 2;  // 单线程单独广播
    }
    else {
        threads = std::min (1024, ceil_pow2 (sizeT));
        blocks = sizeData / sizeT;
        return 1;  // 多线程协作广播
    }
}

#endif

#endif
