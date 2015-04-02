template <class unaryOper, class binaryOper, class Reducer, class DT>
void reduce_1d (DT *odata, const DT *adata, const DT *bdata, const Shape &s, const int keepdim,
    const int maxBlocks, const int maxThreads)
{ const int numData = s.size;
  const int sizeX = s.get_sizeX (keepdim);
  const int strdX = s.get_strdX (keepdim);

  const int numThreads = min (1024, ceilPow2(maxThreads));
//const int numBlocks  = min (maxBlocks, (numData + (2*numThreads - 1)) / (2*numThreads));
  const int smem = numThreads * sizeof(DT);
  dim3 dimGrid  (maxBlocks,  1, 1);
  dim3 dimBlock (numThreads, 1, 1);

  if (s.dims == keepdim || (s.dims == 2 && keepdim == 1))
  switch (dimBlock.x) {
  case 1024: reduce_eq<unaryOper, binaryOper, Reducer, DT,1024><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX);  break;
  case  512: reduce_eq<unaryOper, binaryOper, Reducer, DT, 512><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX);  break;
  case  256: reduce_eq<unaryOper, binaryOper, Reducer, DT, 256><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX);  break;
  case  128: reduce_eq<unaryOper, binaryOper, Reducer, DT, 128><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX);  break;
  case   64: reduce_eq<unaryOper, binaryOper, Reducer, DT,  64><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX);  break;
  case   32: reduce_eq<unaryOper, binaryOper, Reducer, DT,  32><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX);  break;
  case   16: reduce_eq<unaryOper, binaryOper, Reducer, DT,  16><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX);  break;
  case    8: reduce_eq<unaryOper, binaryOper, Reducer, DT,   8><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX);  break;
  case    4: reduce_eq<unaryOper, binaryOper, Reducer, DT,   4><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX);  break;
  case    2: reduce_eq<unaryOper, binaryOper, Reducer, DT,   2><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX);  break;
  case    1: reduce_eq<unaryOper, binaryOper, Reducer, DT,   1><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX);  break;
  }
  else
  switch (dimBlock.x) {
  case 1024: reduce_1d<unaryOper, binaryOper, Reducer, DT,1024><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX, strdX);  break;
  case  512: reduce_1d<unaryOper, binaryOper, Reducer, DT, 512><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX, strdX);  break;
  case  256: reduce_1d<unaryOper, binaryOper, Reducer, DT, 256><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX, strdX);  break;
  case  128: reduce_1d<unaryOper, binaryOper, Reducer, DT, 128><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX, strdX);  break;
  case   64: reduce_1d<unaryOper, binaryOper, Reducer, DT,  64><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX, strdX);  break;
  case   32: reduce_1d<unaryOper, binaryOper, Reducer, DT,  32><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX, strdX);  break;
  case   16: reduce_1d<unaryOper, binaryOper, Reducer, DT,  16><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX, strdX);  break;
  case    8: reduce_1d<unaryOper, binaryOper, Reducer, DT,   8><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX, strdX);  break;
  case    4: reduce_1d<unaryOper, binaryOper, Reducer, DT,   4><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX, strdX);  break;
  case    2: reduce_1d<unaryOper, binaryOper, Reducer, DT,   2><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX, strdX);  break;
  case    1: reduce_1d<unaryOper, binaryOper, Reducer, DT,   1><<<dimGrid, dimBlock, smem>>> (odata, adata, bdata, numData, sizeX, strdX);  break;
  }

  cuda_sync_check ("reduce_1d");
}

template <class unaryOper, class binaryOper, class Saver, class DT>
void broadcast_1d (DT *odata, const DT *adata, const DT *bdata, const Shape &s, const int keepdim,
    const int maxBlocks, const int maxThreads)
{ const int numData = s.size;
  const int sizeX = s.get_sizeX (keepdim);
  const int strdX = s.get_strdX (keepdim);

  const int numThreads = min (1024, ceilPow2(maxThreads));
//const int numBlocks  = min (maxBlocks, (numData + (2*numThreads - 1)) / (2*numThreads));
  dim3 dimGrid  (maxBlocks,  1, 1);
  dim3 dimBlock (numThreads, 1, 1);

  if (s.dims == keepdim || (s.dims == 2 && keepdim == 1))
    broadcast_eq<unaryOper, binaryOper, Saver, DT><<<dimGrid, dimBlock>>> (odata, adata, bdata, numData, sizeX);
  else
    broadcast_1d<unaryOper, binaryOper, Saver, DT><<<dimGrid, dimBlock>>> (odata, adata, bdata, numData, sizeX, strdX);

  cuda_sync_check ("broadcast_1d");
}
