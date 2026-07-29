/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file is_close_dag.h
 * atan2:
 * y = atan2(x1, x2) = atan(x1/x2)  (x2>0)
 *                   = atan(x1/x2) + pi (x2<0 and x1>=0)
 *                   = atan(x1/x2) - pi (x2<0 and x1<0)
 *                   = pi/2 (x2=0 and x1>0)
 *                   = -pi/2 (x2=0 and x1<0)
 *                   = nan (x2=0 and x1=0)
 * atan(x) = x - x^3/3 + x^5/5 - x^7/7 + x^9/9 - x^11/11 + x^13/13... (|x|<=1)
 *         = pi/4 + atan((x-1)/(x+1)) (x>1)
 *         = pi/8 + atan((x-tan(pi/8))/(1+tan(pi/8)*x)) (tan(pi/8)=0.4142135623730950) (x > tan(pi/8) and x <
 * tan(pi/4)))
 */

#ifndef ATAN2_DAG_H
#define ATAN2_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace Atan2Op {
using namespace Ops::Base;
using namespace AscendC;
template <typename T>
__aicore__ constexpr float GetMaxFloat()
{
    if constexpr (std::is_same<T, float>::value) {
        return 10000000.0f;
    }
    return 65504.0f;
}
constexpr float PI = 3.14159265358979323846f;
constexpr float PI_BY_2 = 1.5707963267948966192313216916398f;
constexpr float PI_BY_8 = 0.392699081698724155f;
constexpr float TAN_PI_BY_8 = 0.4142135623730950f;
constexpr float PI_BY_4 = 0.78539816339744831f;
constexpr float PI_3_4 = 2.3561944901923448f;
constexpr uint16_t TAYLOR_SERIES_ORDER = 13;
constexpr uint16_t TAYLOR_SERIES_TERMS = (TAYLOR_SERIES_ORDER + 1) / 2;
constexpr float TAYLOR_SERIES_COEFF[TAYLOR_SERIES_TERMS] = {1.0f,        -1.0f / 3.0f,  1.0f / 5.0f, -1.0f / 7.0f,
                                                            1.0f / 9.0f, -1.0f / 11.0f, 1.0f / 13.0f};
constexpr uint32_t FLOAT_SIGN_MASK = 0x80000000;
constexpr uint32_t FLOAT_ONE_MASK = 0x3F800000;
constexpr float CONST_INF = INFINITY;
constexpr float CONST_NEG_INF = -INFINITY;

template <typename T>
struct Atan2Compute : public Vec::ElemwiseBinaryOP<float, float, float> {
    __aicore__ inline Atan2Compute(
        LocalTensor<float>& dst, LocalTensor<float>& src0, LocalTensor<float>& src1, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint32_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;

        __ubuf__ float* src0Addr = (__ubuf__ float*)src0.GetPhyAddr();
        __ubuf__ float* src1Addr = (__ubuf__ float*)src1.GetPhyAddr();
        __ubuf__ float* dstAddr = (__ubuf__ float*)dst.GetPhyAddr();

        __VEC_SCOPE__
        {
            MicroAPI::MaskReg mask0 = MicroAPI::CreateMask<uint32_t>();
            MicroAPI::RegTensor<uint32_t> floatSignMaskReg;
            MicroAPI::RegTensor<uint32_t> floatOneMaskReg;
            MicroAPI::RegTensor<float> floatZeroReg;
            MicroAPI::RegTensor<float> floatNegZeroReg;
            MicroAPI::RegTensor<float> floatMaxReg;
            MicroAPI::RegTensor<float> piReg;
            MicroAPI::RegTensor<float> negPiReg;
            MicroAPI::RegTensor<float> piBy2Reg;
            MicroAPI::RegTensor<float> negPiBy2Reg;
            MicroAPI::RegTensor<float> piBy4Reg;
            MicroAPI::RegTensor<float> negPiBy4Reg;
            MicroAPI::RegTensor<float> pi3By4Reg;
            MicroAPI::RegTensor<float> negPi3By4Reg;
            MicroAPI::RegTensor<float> adjustReg;
            MicroAPI::Duplicate(floatSignMaskReg, FLOAT_SIGN_MASK, mask0);
            MicroAPI::Duplicate(floatOneMaskReg, FLOAT_ONE_MASK, mask0);
            MicroAPI::Duplicate(floatZeroReg, 0.0f, mask0);
            MicroAPI::Duplicate(floatMaxReg, GetMaxFloat<float>(), mask0);
            MicroAPI::Duplicate(piReg, PI, mask0);
            MicroAPI::Duplicate(negPiReg, -PI, mask0);
            MicroAPI::Duplicate(piBy2Reg, PI_BY_2, mask0);
            MicroAPI::Duplicate(negPiBy2Reg, -PI_BY_2, mask0);
            MicroAPI::Duplicate(piBy4Reg, PI_BY_4, mask0);
            MicroAPI::Duplicate(negPiBy4Reg, -PI_BY_4, mask0);
            MicroAPI::Duplicate(pi3By4Reg, PI_3_4, mask0);
            MicroAPI::Duplicate(negPi3By4Reg, -PI_3_4, mask0);

            MicroAPI::RegTensor<float> addReg1; // 1.0f
            MicroAPI::RegTensor<float> addReg3; // -1.0f / 3.0f
            MicroAPI::RegTensor<float> addReg5; // 1.0f / 5.0f
            MicroAPI::RegTensor<float> addReg7; // -1.0f / 7.0f
            MicroAPI::RegTensor<float> addReg9; // 1.0f / 9.0f
            MicroAPI::Duplicate(addReg1, 1.0f, mask0);
            MicroAPI::Duplicate(addReg3, -1.0f / 3.0f, mask0);
            MicroAPI::Duplicate(addReg5, 1.0f / 5.0f, mask0);
            MicroAPI::Duplicate(addReg7, -1.0f / 7.0f, mask0);
            MicroAPI::Duplicate(addReg9, 1.0f / 9.0f, mask0);

            MicroAPI::RegTensor<float> x1Reg;
            MicroAPI::RegTensor<float> x2Reg;
            MicroAPI::RegTensor<float> yReg;
            MicroAPI::RegTensor<float> xReg;
            MicroAPI::MaskReg x1Mask;
            MicroAPI::MaskReg x2Mask;
            MicroAPI::MaskReg yMask;
            MicroAPI::MaskReg mask;

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                MicroAPI::Duplicate(adjustReg, 0.0f, mask0);
                mask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(count);
                MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(x1Reg, src0Addr, vlSize);
                MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(x2Reg, src1Addr, vlSize);
                MicroAPI::Div(xReg, x1Reg, x2Reg, mask); // x1/x2
                Atan(yReg, xReg, floatSignMaskReg, floatOneMaskReg, floatMaxReg,
                    addReg1, addReg3, addReg5, addReg7, addReg9, mask);
                // add pi where x2<0 and x1>=0, -pi where x2<0 and x1<0
                MicroAPI::CompareScalar<float, CMPMODE::LT>(x2Mask, x2Reg, 0.0f, mask); // x2 < 0
                MicroAPI::CompareScalar<float, CMPMODE::GE>(x1Mask, x1Reg, 0.0f, mask); // x1 >= 0
                MicroAPI::MaskAnd(yMask, x1Mask, x2Mask, mask);                         // x1 >=0 && x2 < 0
                MicroAPI::Select(adjustReg, piReg, adjustReg, yMask);
                MicroAPI::MaskNot(x1Mask, x1Mask, mask);        // x1 < 0
                MicroAPI::MaskAnd(yMask, x1Mask, x2Mask, mask); // x1 < 0 && x2 < 0
                MicroAPI::Select(adjustReg, negPiReg, adjustReg, yMask);
                MicroAPI::Add(yReg, yReg, adjustReg, mask);
                PostProcess(yReg, x1Reg, x2Reg, yMask, mask, floatSignMaskReg, floatOneMaskReg, floatZeroReg,
                    piReg, negPiReg, piBy2Reg, negPiBy2Reg, piBy4Reg, negPiBy4Reg, pi3By4Reg, negPi3By4Reg);
                MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, yReg, vlSize, mask);
            }
        }
    }

    __aicore__ inline void  PostProcess(
        MicroAPI::RegTensor<float>& yReg, MicroAPI::RegTensor<float>& x1Reg,
        MicroAPI::RegTensor<float>& x2Reg, MicroAPI::MaskReg& yMask,
        MicroAPI::MaskReg& mask, MicroAPI::RegTensor<uint32_t>& floatSignMaskReg,
        MicroAPI::RegTensor<uint32_t>& floatOneMaskReg, MicroAPI::RegTensor<float>& floatZeroReg,
        MicroAPI::RegTensor<float>& piReg, MicroAPI::RegTensor<float>& negPiReg,
        MicroAPI::RegTensor<float>& piBy2Reg, MicroAPI::RegTensor<float>& negPiBy2Reg,
        MicroAPI::RegTensor<float>& piBy4Reg, MicroAPI::RegTensor<float>& negPiBy4Reg,
        MicroAPI::RegTensor<float>& pi3By4Reg, MicroAPI::RegTensor<float>& negPi3By4Reg)
    {
        MicroAPI::RegTensor<float> xSignReg;
        MicroAPI::MaskReg x1ZeroMask;
        MicroAPI::MaskReg x1NegMask;
        MicroAPI::MaskReg x1PosZeroMask;
        MicroAPI::MaskReg x1NegZeroMask;
        MicroAPI::MaskReg x1PosInfMask;
        MicroAPI::MaskReg x1NegInfMask;

        MicroAPI::MaskReg x2ZeroMask;
        MicroAPI::MaskReg x2NegMask;
        MicroAPI::MaskReg x2PosMask;
        MicroAPI::MaskReg x2NegZeroMask;
        MicroAPI::MaskReg x2PosInfMask;
        MicroAPI::MaskReg x2NegInfMask;

        MicroAPI::CompareScalar<float, CMPMODE::EQ>(x1ZeroMask, x1Reg, 0.0f, mask); // x1 = ±0.0
        MicroAPI::And(
            (MicroAPI::RegTensor<uint32_t>&)xSignReg, (MicroAPI::RegTensor<uint32_t>&)x1Reg, floatSignMaskReg,
            mask); // 取x1符号位
        MicroAPI::Or(
            (MicroAPI::RegTensor<uint32_t>&)xSignReg, (MicroAPI::RegTensor<uint32_t>&)xSignReg, floatOneMaskReg,
            mask); // +/- 1
        MicroAPI::CompareScalar<float, CMPMODE::EQ>(x1NegMask, xSignReg, -1.0f, mask); // x1 is neg
        MicroAPI::MaskAnd(x1NegZeroMask, x1ZeroMask, x1NegMask, mask);  // x1 = -0.0
        MicroAPI::MaskXor(x1PosZeroMask, x1ZeroMask, x1NegZeroMask, mask);  // x1 = +0.0
        
        MicroAPI::CompareScalar<float, CMPMODE::EQ>(x2ZeroMask, x2Reg, 0.0f, mask); // x2 = ±0.0
        MicroAPI::And(
            (MicroAPI::RegTensor<uint32_t>&)xSignReg, (MicroAPI::RegTensor<uint32_t>&)x2Reg, floatSignMaskReg,
            mask); // 取x2符号位
        MicroAPI::Or(
            (MicroAPI::RegTensor<uint32_t>&)xSignReg, (MicroAPI::RegTensor<uint32_t>&)xSignReg, floatOneMaskReg,
            mask); // +/- 1
        MicroAPI::CompareScalar<float, CMPMODE::EQ>(x2NegMask, xSignReg, -1.0f, mask); // x2 is neg
        MicroAPI::MaskNot(x2PosMask, x2NegMask, mask); // x2 is pos
        MicroAPI::MaskAnd(x2NegZeroMask, x2ZeroMask, x2NegMask, mask);  // x2 = -0.0 

        // (x1 < 0 || x1 = -0.0) && x2 = -0.0 --> -pi/2; x1 = -0.0后续修正
        MicroAPI::MaskAnd(yMask, x1NegMask, x2NegZeroMask, mask);
        MicroAPI::Select(yReg, negPiBy2Reg, yReg, yMask);

        // (x1 > 0 || x1 = +0.0) && x2 = -0.0 --> pi/2; x1 = 0.0后续修正
        MicroAPI::MaskNot(yMask, x1NegMask, mask); // x1 is pos
        MicroAPI::MaskAnd(yMask, yMask, x2NegZeroMask, mask);
        MicroAPI::Select(yReg, piBy2Reg, yReg, yMask);
        
        // x1 = -0.0 && (x2 < 0.0 || x2 = -0.0) --> -pi
        MicroAPI::MaskAnd(yMask, x1NegZeroMask, x2NegMask, mask);
        MicroAPI::Select(yReg, negPiReg, yReg, yMask);

        // x1 = -0.0 && (x2 > 0.0 || x2 = +0.0) --> -0.0
        MicroAPI::MaskAnd(yMask, x1NegZeroMask, x2PosMask, mask);
        MicroAPI::Select(yReg, (MicroAPI::RegTensor<float>&)floatSignMaskReg, yReg, yMask);

        // x1 = 0.0 && (x2 < 0.0 || x2 = -0.0) --> pi
        MicroAPI::MaskAnd(yMask, x1PosZeroMask, x2NegMask, mask);
        MicroAPI::Select(yReg, piReg, yReg, yMask);

        // x1 = 0.0 && (x2 > 0.0 || x2 = +0.0) --> 0.0
        MicroAPI::MaskAnd(yMask, x1PosZeroMask, x2PosMask, mask);
        MicroAPI::Select(yReg, floatZeroReg, yReg, yMask);

        MicroAPI::CompareScalar<float, CMPMODE::EQ>(x1PosInfMask, x1Reg, CONST_INF, mask); // x1 = +inf
        MicroAPI::CompareScalar<float, CMPMODE::EQ>(x1NegInfMask, x1Reg, CONST_NEG_INF, mask); // x1 = -inf
        MicroAPI::CompareScalar<float, CMPMODE::EQ>(x2PosInfMask, x2Reg, CONST_INF, mask); // x2 = +inf
        MicroAPI::CompareScalar<float, CMPMODE::EQ>(x2NegInfMask, x2Reg, CONST_NEG_INF, mask); // x2 = -inf

        // x1 < 0 && x2 = inf --> -0.0 --> -0.0; x1 = -inf后续修正
        MicroAPI::MaskAnd(yMask, x1NegMask, x2PosInfMask, mask);
        MicroAPI::Select(yReg, (MicroAPI::RegTensor<float>&)floatSignMaskReg, yReg, yMask);

        // x1 = inf && x2 = inf --> pi/4
        MicroAPI::MaskAnd(yMask, x1PosInfMask, x2PosInfMask, mask);
        MicroAPI::Select(yReg, piBy4Reg, yReg, yMask);

        // x1 = -inf && x2 = inf --> -pi/4
        MicroAPI::MaskAnd(yMask, x1NegInfMask, x2PosInfMask, mask);
        MicroAPI::Select(yReg, negPiBy4Reg, yReg, yMask);

        // x1 = inf && x2 = -inf --> 3pi/4
        MicroAPI::MaskAnd(yMask, x1PosInfMask, x2NegInfMask, mask);
        MicroAPI::Select(yReg, pi3By4Reg, yReg, yMask);

        // x1 = -inf && x2 = -inf --> -3pi/4
        MicroAPI::MaskAnd(yMask, x1NegInfMask, x2NegInfMask, mask);
        MicroAPI::Select(yReg, negPi3By4Reg, yReg, yMask);
    }

    __aicore__ inline void TaylorSeriesExpansion(
        MicroAPI::RegTensor<float>& yReg, MicroAPI::RegTensor<float>& xReg,
        MicroAPI::RegTensor<float>& addReg1, MicroAPI::RegTensor<float>& addReg3,
        MicroAPI::RegTensor<float>& addReg5, MicroAPI::RegTensor<float>& addReg7,
        MicroAPI::RegTensor<float>& addReg9, MicroAPI::MaskReg& mask)
    {
        // x - 1/3*x^3 + 1/5*x^5 - 1/7*x^7 + 1/9*x^9 - 1/11*x^11 + 1/13*x^13
        // x(1 - x^2(1/3 - x^2(1/5 - x^2(1/7 - x^2(1/9 - x^2(1/11 - x^2/13))))))
        MicroAPI::RegTensor<float> xSquarReg;
        MicroAPI::Mul(xSquarReg, xReg, xReg, mask);

        MicroAPI::Duplicate(yReg, -1.0f / 11.0f, mask);
        MicroAPI::Axpy(yReg, xSquarReg, 1.0f / 13.0f, mask);
        MicroAPI::MulDstAdd(yReg, xSquarReg, addReg9, mask);
        MicroAPI::MulDstAdd(yReg, xSquarReg, addReg7, mask);
        MicroAPI::MulDstAdd(yReg, xSquarReg, addReg5, mask);
        MicroAPI::MulDstAdd(yReg, xSquarReg, addReg3, mask);
        MicroAPI::MulDstAdd(yReg, xSquarReg, addReg1, mask);
        MicroAPI::Mul(yReg, yReg, xReg, mask);
    }

    __aicore__ inline void AtanLT1(
        MicroAPI::RegTensor<float>& yReg, MicroAPI::RegTensor<float>& xReg,
        MicroAPI::RegTensor<float>& addReg1, MicroAPI::RegTensor<float>& addReg3,
        MicroAPI::RegTensor<float>& addReg5, MicroAPI::RegTensor<float>& addReg7,
        MicroAPI::RegTensor<float>& addReg9, MicroAPI::MaskReg& mask)
    {
        MicroAPI::RegTensor<float> tmpReg0;
        MicroAPI::RegTensor<float> tmpReg1;
        MicroAPI::Muls(tmpReg0, xReg, TAN_PI_BY_8, mask);
        MicroAPI::Adds(tmpReg0, tmpReg0, 1.0f, mask);
        MicroAPI::Adds(tmpReg1, xReg, -TAN_PI_BY_8, mask);
        MicroAPI::Div(tmpReg0, tmpReg1, tmpReg0, mask);
        MicroAPI::Abs(tmpReg1, tmpReg0, mask);
        TaylorSeriesExpansion(tmpReg0, tmpReg1, addReg1, addReg3, addReg5, addReg7, addReg9, mask);
        MicroAPI::Adds(tmpReg0, tmpReg0, PI_BY_8, mask);
        TaylorSeriesExpansion(tmpReg1, xReg, addReg1, addReg3, addReg5, addReg7, addReg9, mask);
        MicroAPI::Min(yReg, tmpReg0, tmpReg1, mask);
    }

    __aicore__ inline void Atan(
        MicroAPI::RegTensor<float>& yReg, MicroAPI::RegTensor<float>& xReg,
        MicroAPI::RegTensor<uint32_t>& floatSignMaskReg, MicroAPI::RegTensor<uint32_t>& floatOneMaskReg,
        MicroAPI::RegTensor<float>& thresholdReg, MicroAPI::RegTensor<float>& addReg1,
        MicroAPI::RegTensor<float>& addReg3, MicroAPI::RegTensor<float>& addReg5,
        MicroAPI::RegTensor<float>& addReg7, MicroAPI::RegTensor<float>& addReg9,
        MicroAPI::MaskReg& mask)
    {
        MicroAPI::RegTensor<float> absReg;
        MicroAPI::RegTensor<float> tmpReg0;
        MicroAPI::RegTensor<float> tmpReg1;
        MicroAPI::RegTensor<float> signReg;
        MicroAPI::And(
            (MicroAPI::RegTensor<uint32_t>&)signReg, (MicroAPI::RegTensor<uint32_t>&)xReg, floatSignMaskReg,
            mask); // 取x2符号位
        MicroAPI::Or(
            (MicroAPI::RegTensor<uint32_t>&)signReg, (MicroAPI::RegTensor<uint32_t>&)signReg, floatOneMaskReg,
            mask); // +/- 1
        MicroAPI::Abs(absReg, xReg, mask);
        MicroAPI::Min(absReg, absReg, thresholdReg, mask); // avoid overflow

        MicroAPI::Adds(tmpReg0, absReg, -1.0f, mask);
        MicroAPI::Adds(tmpReg1, absReg, 1.0f, mask);
        MicroAPI::Div(tmpReg0, tmpReg0, tmpReg1, mask);
        MicroAPI::Abs(tmpReg1, tmpReg0, mask);
        AtanLT1(tmpReg0, tmpReg1, addReg1, addReg3, addReg5, addReg7, addReg9, mask);
        MicroAPI::Adds(tmpReg0, tmpReg0, PI_BY_4, mask);
        AtanLT1(tmpReg1, absReg, addReg1, addReg3, addReg5, addReg7, addReg9, mask);
        MicroAPI::Min(yReg, tmpReg0, tmpReg1, mask);
        // recover sign
        MicroAPI::Mul(yReg, yReg, signReg, mask);
#endif
    }
};

template <typename T>
struct Atan2Dag {
    using InputX1T = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2T = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    // cast
    using InputX1 = Bind<Vec::Cast<float, T, 0>, InputX1T>;
    using InputX2 = Bind<Vec::Cast<float, T, 0>, InputX2T>;

    using Atan2Res = Bind<Atan2Compute<T>, InputX1, InputX2>;
    using OpCastRes = Bind<Vec::Cast<T, float, 1>, Atan2Res>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCastRes>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace Atan2Op

#endif // ATAN2_DAG_H