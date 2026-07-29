#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
__aicore__ inline bool IsGreater(T a, T b) { return a > b; }
template <>
__aicore__ inline bool IsGreater<half>(half a, half b) { return static_cast<float>(a) > static_cast<float>(b); }
template <>
__aicore__ inline bool IsGreater<bfloat16_t>(bfloat16_t a, bfloat16_t b) { return static_cast<float>(a) > static_cast<float>(b); }

template <typename T>
class KernelGreaterS9 {
public:
 __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR y, const GreaterTilingData &t) {
    a_.SetGlobalBuffer((__gm__ T *)a); b_.SetGlobalBuffer((__gm__ T *)b);
    y_.SetGlobalBuffer((__gm__ int8_t *)y);
    rank_ = t.rank; blocks_ = t.blockDim; total_ = t.outputCount;
    for (uint32_t i = 0; i < rank_; ++i) { shape_[i] = t.outputShape[i]; sa_[i] = t.x1Strides[i]; sb_[i] = t.x2Strides[i]; }
 }
 __aicore__ inline void Process() {
    for (uint64_t idx = GetBlockIdx(); idx < total_; idx += blocks_) {
        uint64_t rem = idx, oa = 0, ob = 0;
        for (int32_t d = rank_ - 1; d >= 0; --d) { const uint64_t c = rem % shape_[d]; rem /= shape_[d]; oa += c * sa_[d]; ob += c * sb_[d]; }
        y_.SetValue(idx, IsGreater<T>(a_.GetValue(oa), b_.GetValue(ob)) ? 1 : 0);
    }
 }
private:
 GlobalTensor<T> a_, b_; GlobalTensor<int8_t> y_;
 uint64_t shape_[8], sa_[8], sb_[8], total_; uint32_t rank_, blocks_;
};
template <typename T> __aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR y, const GreaterTilingData &t) { KernelGreaterS9<T> op; op.Init(a,b,y,t); op.Process(); }
extern "C" __global__ __aicore__ void greater(GM_ADDR a, GM_ADDR b, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
 GET_TILING_DATA(t, tiling);
 if (t.dtypeCode == 0) Run<half>(a,b,y,t); else if (t.dtypeCode == 1) Run<float>(a,b,y,t);
 else if (t.dtypeCode == 2) Run<int32_t>(a,b,y,t); else if (t.dtypeCode == 3) Run<int8_t>(a,b,y,t); else Run<bfloat16_t>(a,b,y,t);
}
