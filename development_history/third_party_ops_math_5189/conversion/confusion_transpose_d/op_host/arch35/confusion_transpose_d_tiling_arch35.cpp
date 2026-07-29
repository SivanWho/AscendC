/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file confusion_transpose_d_tiling_arch35.cpp
 * \brief
 */

#include "confusion_transpose_d_tiling_arch35.h"
#include "../../../transpose/op_host/arch35/transpose_tiling_arch35.h"

namespace optiling {

static const std::vector<ge::DataType> SUPPORT_DTYPE_ND = {ge::DT_INT8,    ge::DT_INT16,  ge::DT_INT32,  ge::DT_INT64,
                                                           ge::DT_UINT8,   ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64,
                                                           ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_BF16};

static const std::vector<ge::DataType> SUPPORT_DTYPE_NZ = {ge::DT_INT8,    ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT32,
                                                           ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};

int64_t ConfusionTransposeDTiling::Prod(vector<int64_t>& inputShape)
{
    int64_t result = 1;
    for (auto i : inputShape) {
        result *= i;
    }
    return result;
}

vector<int64_t> ConfusionTransposeDTiling::ReshapeFrac(vector<int64_t>& shapeIn, vector<int64_t>& shapeOut)
{
    if (Prod(shapeIn) != Prod(shapeOut)) {
        throw std::invalid_argument("non-equal products!");
    }

    int64_t idxIn = shapeIn.size() - 1;
    int64_t idxOut = shapeOut.size() - 1;
    vector<int64_t> shapeFrac;
    int64_t resIn = shapeIn[idxIn];
    int64_t resOut = shapeOut[idxOut];

    while (idxIn >= 0 || idxOut >= 0) {
        int fracElmt = std::min(resIn, resOut);
        shapeFrac.insert(shapeFrac.begin(), fracElmt);

        if (resIn % fracElmt != 0 || resOut % fracElmt != 0) {
            std::string msg =
                "The input and output of reshape should be able to be obtained through factorization and\
reorganization. Error shape is " +
                VectorToString(shapeIn) + " and " + VectorToString(shapeOut);
            throw std::invalid_argument(msg.c_str());
        }

        resIn /= fracElmt;
        resOut /= fracElmt;

        if (resIn <= 1) {
            idxIn -= 1;
            if (idxIn >= 0) {
                resIn = shapeIn[idxIn];
            }
        }
        if (resOut <= 1) {
            idxOut -= 1;
            if (idxOut >= 0) {
                resOut = shapeOut[idxOut];
            }
        }
    }

    return shapeFrac;
}

std::string ConfusionTransposeDTiling::VectorToString(vector<vector<int64_t>>& vec)
{
    std::stringstream ss;
    ss << "[ ";
    for (int i = 0; i < (int64_t)vec.size(); i++) {
        if (i) {
            ss << ", ";
        }
        ss << "[ ";
        for (int j = 0; j < (int64_t)vec[i].size(); j++) {
            if (j) {
                ss << ", ";
            }
            ss << vec[i][j];
        }

        ss << "] ";
    }
    ss << "] ";
    return ss.str();
}

std::string ConfusionTransposeDTiling::VectorToString(vector<int64_t>& vec)
{
    std::stringstream ss;
    ss << "[ ";
    for (int i = 0; i < (int64_t)vec.size(); i++) {
        if (i) {
            ss << ", ";
        }
        ss << vec[i];
    }
    ss << "] ";
    return ss.str();
}

vector<vector<int64_t>> ConfusionTransposeDTiling::SplitShapeIn(
    vector<vector<int64_t>>& inputShape, vector<int64_t>& axisList)
{
    vector<vector<int64_t>> inputShapeCopy(inputShape);
    for (auto i : axisList) {
        int64_t index = inputShapeCopy.size() + i;
        if (inputShapeCopy[index].size() == 1) {
            if (inputShapeCopy[index][0] < SPLIT_NUM) {
                std::string msg = "the splited dim(usually the last dim) should not be smaller than 16. error shape is: " +
                             VectorToString(inputShape);
                throw std::invalid_argument(msg.c_str());
            }
            inputShapeCopy[index].insert(inputShapeCopy[index].begin(), inputShapeCopy[index][0] / SPLIT_NUM);
            inputShapeCopy[index][1] = SPLIT_NUM;
        } else {
            if (inputShapeCopy[index].back() > SPLIT_NUM) {
                inputShapeCopy[index].insert(inputShapeCopy[index].end() - 1, inputShapeCopy[index].back() / SPLIT_NUM);
                inputShapeCopy[index].back() = SPLIT_NUM;
            }
        }
    }

    return inputShapeCopy;
}

vector<vector<int64_t>> ConfusionTransposeDTiling::MergeFrac(vector<int64_t>& shapeTensor, vector<int64_t>& fracShape)
{
    if (shapeTensor.size() == fracShape.size()) {
        vector<vector<int64_t>> fracShapeNested;
        for (auto i : fracShape) {
            fracShapeNested.push_back({i});
        }
        return fracShapeNested;
    }
    if (std::find(shapeTensor.begin(), shapeTensor.end(), 1) == shapeTensor.end()) {
        return MergeFrac1(shapeTensor, fracShape);
    }

    return MergeFrac2(shapeTensor, fracShape);
}

vector<vector<int64_t>> ConfusionTransposeDTiling::MergeFrac1(vector<int64_t>& shapeTensor, vector<int64_t>& fracShape)
{
    int64_t tensorIdx = shapeTensor.size() - 1;
    int64_t fracIdx = fracShape.size() - 1;

    vector<vector<int64_t>> mergedList;
    while (tensorIdx >= 0) {
        int64_t outElmt = shapeTensor[tensorIdx];
        vector<int64_t> shapeList = {fracShape[fracIdx]};
        int64_t listProduct = fracShape[fracIdx];
        while (listProduct < outElmt || (fracIdx > 0 && fracShape[fracIdx - 1] == 1)) {
            fracIdx--;
            shapeList.insert(shapeList.begin(), fracShape[fracIdx]);
            listProduct *= fracShape[fracIdx];
        }
        mergedList.insert(mergedList.begin(), shapeList);
        tensorIdx--;
        fracIdx--;
    }
    while (fracIdx >= 0) {
        mergedList.insert(mergedList.begin(), {fracShape[fracIdx]});
        fracIdx--;
    }
    return mergedList;
}

bool ConfusionTransposeDTiling::MergeFrac2Judge(
    vector<int64_t>& shapeTensor, vector<int64_t>& fracShape, vector<int64_t>& shapeList, int64_t& tensorIdx,
    int64_t& fracIdx, int64_t& listProduct, int64_t& outElmt)
{
    if (shapeTensor[tensorIdx - 1] == 1 && fracShape[fracIdx - 1] == 1) {
        if (fracShape[fracIdx] == 1) {
            return false;
        }
        if (listProduct == outElmt) {
            if (fracIdx > 1 && fracShape[fracIdx - DIM_TWO] == 1) {
                fracIdx--;
                shapeList.insert(shapeList.begin(), fracShape[fracIdx]);
                listProduct *= fracShape[fracIdx];
            }
            return false;
        }
    }
    return true;
}

vector<vector<int64_t>> ConfusionTransposeDTiling::MergeFrac2(vector<int64_t>& shapeTensor, vector<int64_t>& fracShape)
{
    int64_t tensorIdx = shapeTensor.size() - 1;
    int64_t fracIdx = fracShape.size() - 1;
    vector<vector<int64_t>> mergedList;
    while (tensorIdx >= 0) {
        int64_t outElmt = shapeTensor[tensorIdx];
        vector<int64_t> shapeList = {fracShape[fracIdx]};
        int64_t listProduct = fracShape[fracIdx];
        while (listProduct < outElmt || (fracIdx > 0 && fracShape[fracIdx - 1] == 1)) {
            if (fracShape[fracIdx] == 1 && shapeTensor[tensorIdx] == 1) {
                break;
            }
            if (std::count(fracShape.begin(), fracShape.begin() + fracIdx, 1) ==
                    std::count(shapeTensor.begin(), shapeTensor.begin() + tensorIdx, 1) &&
                listProduct == outElmt) {
                break;
            }
            if (!MergeFrac2Judge(shapeTensor, fracShape, shapeList, tensorIdx, fracIdx, listProduct, outElmt)) {
                break;
            }
            fracIdx -= 1;
            shapeList.insert(shapeList.begin(), fracShape[fracIdx]);
            listProduct *= fracShape[fracIdx];
            if (fracIdx == 0) {
                break;
            }
        }
        mergedList.insert(mergedList.begin(), shapeList);
        tensorIdx--;
        fracIdx--;
    }
    while (fracIdx >= 0) {
        mergedList.insert(mergedList.begin(), {fracShape[fracIdx]});
        fracIdx--;
    }
    return mergedList;
}

vector<vector<int64_t>> ConfusionTransposeDTiling::MergePerm(vector<int64_t>& srcList, vector<vector<int64_t>>& dstList)
{
    vector<int64_t> lenList;
    vector<vector<int64_t>> resultList;

    int64_t index = 0;
    for (int64_t i = 0; i < int64_t(dstList.size()); i++) {
        resultList.push_back(vector<int64_t>());
        for (int64_t j = 0; j < int64_t(dstList[i].size()); j++) {
            resultList[i].push_back(srcList[index++]);
        }
    }

    return resultList;
}

vector<int64_t> ConfusionTransposeDTiling::FlatPerm(vector<vector<int64_t>>& mergedPerm)
{
    vector<int64_t> flatenPerm;
    for (int64_t i = 0; i < int64_t(mergedPerm.size()); i++) {
        for (int64_t j = 0; j < int64_t(mergedPerm[i].size()); j++) {
            flatenPerm.push_back(mergedPerm[i][j]);
        }
    }

    return flatenPerm;
}

vector<int64_t> ConfusionTransposeDTiling::PermOrigToNz(vector<vector<int64_t>>& permMerged, int64_t splitIdx)
{
    vector<int64_t> permToNz;
    for (int64_t i = 0; i < int64_t(permMerged.size()) - DIM_TWO; i++) {
        permToNz.insert(permToNz.end(), permMerged[i].begin(), permMerged[i].end());
    }
    permToNz.insert(permToNz.end(), permMerged.back().begin(), permMerged.back().end() + splitIdx);
    permToNz.insert(
        permToNz.end(), permMerged[permMerged.size() - DIM_TWO].begin(), permMerged[permMerged.size() - DIM_TWO].end());
    permToNz.insert(permToNz.end(), permMerged.back().end() + splitIdx, permMerged.back().end());

    return permToNz;
}

void ConfusionTransposeDTiling::PermNzToOrig(
    vector<vector<int64_t>>& mergedOrig, int64_t splitIdx, vector<int64_t>& permToOrig, vector<int64_t>& fracNzIn)
{
    int64_t cnt = 0;
    for (int64_t i = 0; i < int64_t(mergedOrig.size()) - DIM_TWO; i++) {
        const auto& frac = mergedOrig[i];
        for (int64_t j = 0; j < int64_t(frac.size()); j++) {
            permToOrig.push_back(cnt);
            cnt++;
            fracNzIn.push_back(frac[j]);
        }
    }

    int64_t cntStart = cnt;
    const auto lastSecondFrac = mergedOrig[mergedOrig.size() - DIM_TWO];
    const auto lastFrac = mergedOrig[mergedOrig.size() - 1];
    for (int64_t i = 0; i < int64_t(lastSecondFrac.size()); i++) {
        permToOrig.push_back(cnt + lastFrac.size() + splitIdx);
        fracNzIn.push_back(lastSecondFrac[i]);
        cnt++;
    }

    for (int64_t i = 0; i < int64_t(lastFrac.size()) + splitIdx; i++) {
        permToOrig.push_back(cntStart);
        fracNzIn.insert(fracNzIn.begin() + cntStart, lastFrac[i]);
        cntStart++;
        cnt++;
    }

    for (int64_t i = int64_t(lastFrac.size()) + splitIdx; i < int64_t(lastFrac.size()); i++) {
        permToOrig.push_back(cnt);
        fracNzIn.push_back(lastFrac[i]);
        cnt++;
    }
}

vector<vector<int64_t>> ConfusionTransposeDTiling::ShapeAfterTranspose(
    vector<vector<int64_t>>& inputShape, vector<int64_t>& transPerm)
{
    if (inputShape.size() != transPerm.size()) {
        throw std::invalid_argument("shape not equal!");
    }
    vector<vector<int64_t>> transposeMergedPerm;
    for (int64_t i = 0; i < int64_t(inputShape.size()); i++) {
        transposeMergedPerm.push_back(vector<int64_t>());
    }
    for (int64_t i = 0; i < int64_t(inputShape.size()); i++) {
        transposeMergedPerm[i] = inputShape[transPerm[i]];
    }
    return transposeMergedPerm;
}

vector<int64_t> ConfusionTransposeDTiling::ShapeAfterTranspose(vector<int64_t>& inputShape, vector<int64_t>& transPerm)
{
    if (inputShape.size() != transPerm.size()) {
        throw std::invalid_argument("shape not equal!");
    }
    vector<int64_t> transposeMergedPerm = inputShape;

    for (int64_t i = 0; i < int64_t(inputShape.size()); i++) {
        transposeMergedPerm[i] = inputShape[transPerm[i]];
    }
    return transposeMergedPerm;
}

vector<vector<int64_t>> ConfusionTransposeDTiling::ShapeBeforeTranspose(
    vector<vector<int64_t>>& mergedFrac, vector<int64_t>& transposePerm)
{
    if (mergedFrac.size() != transposePerm.size()) {
        throw std::invalid_argument("shape not equal!");
    }
    vector<vector<int64_t>> shapeBefore;
    for (int64_t i = 0; i < int64_t(mergedFrac.size()); i++) {
        shapeBefore.push_back(vector<int64_t>());
    }
    for (int64_t i = 0; i < int64_t(mergedFrac.size()); i++) {
        shapeBefore[transposePerm[i]] = mergedFrac[i];
    }
    return shapeBefore;
}

void ConfusionTransposeDTiling::TransposeReshape(
    vector<int64_t>& transposePerm, vector<int64_t>& reshapeIn, vector<int64_t>& reshapeOut, vector<int64_t>& fracNzIn,
    vector<int64_t>& finalPerm)
{
    vector<int64_t> fracRes = ReshapeFrac(reshapeOut, reshapeIn);
    vector<vector<int64_t>> mergeFracOut = MergeFrac(reshapeOut, fracRes);

    vector<int64_t> tmp({-1, -2});
    mergeFracOut = SplitShapeIn(mergeFracOut, tmp);

    vector<int64_t> mergeFracOutFlat = FlatPerm(mergeFracOut);
    vector<vector<int64_t>> mergeFracIn = MergeFrac(reshapeIn, mergeFracOutFlat);
    vector<vector<int64_t>> fracMerged = ShapeBeforeTranspose(mergeFracIn, transposePerm);
    vector<vector<int64_t>> transIn = SplitShapeIn(fracMerged, tmp);
    vector<vector<int64_t>> mergedFracInSplit = ShapeAfterTranspose(transIn, transposePerm);
    vector<int64_t> mergedFracInSplitFlat = FlatPerm(mergedFracInSplit);
    vector<vector<int64_t>> mergedFracOutSplit = MergeFrac(reshapeOut, mergedFracInSplitFlat);
    vector<int64_t> nzNdPerm;
    PermNzToOrig(transIn, -1, nzNdPerm, fracNzIn);
    vector<vector<int64_t>> permTransIn = MergePerm(nzNdPerm, transIn);
    vector<vector<int64_t>> permTranspose = ShapeAfterTranspose(permTransIn, transposePerm);
    vector<int64_t> permTransposeFlat = FlatPerm(permTranspose);
    vector<vector<int64_t>> permMerged = MergePerm(permTransposeFlat, mergedFracOutSplit);

    finalPerm = PermOrigToNz(permMerged, -1);
}

void ConfusionTransposeDTiling::ReshapeTranspose(
    vector<int64_t>& transposePerm, vector<int64_t>& reshapeIn, vector<int64_t>& reshapeOut, vector<int64_t>& fracNzIn,
    vector<int64_t>& finalPerm)
{
    vector<int64_t> transOut = ShapeAfterTranspose(reshapeOut, transposePerm);
    int64_t idx = transOut.size();
    vector<vector<int64_t>> transOutMerge;
    for (int64_t i = 0; i < idx; i++) {
        transOutMerge.push_back(vector<int64_t>());
    }
    for (int64_t i = 0; i < idx; i++) {
        transOutMerge[i].push_back(transOut[i]);
    }

    vector<int64_t> tmp({-1, -2});
    vector<vector<int64_t>> transOutSplit = SplitShapeIn(transOutMerge, tmp);
    vector<vector<int64_t>> mergedFrac = ShapeBeforeTranspose(transOutSplit, transposePerm);
    vector<int64_t> mergedFracFlat = FlatPerm(mergedFrac);
    vector<int64_t> fracRes = ReshapeFrac(mergedFracFlat, reshapeIn);
    vector<vector<int64_t>> mergedFracIn = MergeFrac(reshapeIn, fracRes);
    vector<vector<int64_t>> mergedFracInSplit = SplitShapeIn(mergedFracIn, tmp);
    vector<int64_t> mergedFracInSplitFlat = FlatPerm(mergedFracInSplit);
    vector<vector<int64_t>> mergedFracSplit = MergeFrac(reshapeOut, mergedFracInSplitFlat);

    vector<int64_t> nzNdPerm;
    PermNzToOrig(mergedFracInSplit, -1, nzNdPerm, fracNzIn);
    vector<vector<int64_t>> permMerged = MergePerm(nzNdPerm, mergedFracSplit);
    vector<vector<int64_t>> transPerm = ShapeAfterTranspose(permMerged, transposePerm);
    finalPerm = PermOrigToNz(transPerm, -1);
}

vector<int64_t> ConfusionTransposeDTiling::TransShapeToVector(gert::Shape inShape)
{
    vector<int64_t> outVector;

    int64_t shapeSize = inShape.GetDimNum();
    for (int64_t i = 0; i < shapeSize; i++) {
        outVector.push_back(inShape[i]);
    }

    return outVector;
}

ge::graphStatus ConfusionTransposeDTiling::GetParameters()
{
    auto xStorageShape = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xStorageShape);
    auto xShape = xStorageShape->GetStorageShape();
    paramInfo_.xShape = xShape;

    auto xDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xDesc);
    auto xFormat = xDesc->GetFormat().GetStorageFormat();
    auto xDtype = xDesc->GetDataType();
    paramInfo_.xDtype = xDtype;
    paramInfo_.xFormat = xFormat;

    auto yStorageShape = tilingContext_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, yStorageShape);
    auto yShape = yStorageShape->GetStorageShape();
    paramInfo_.yShape = yShape;

    auto yDesc = tilingContext_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, yDesc);
    auto yDtype = yDesc->GetDataType();
    auto yFormat = yDesc->GetFormat().GetStorageFormat();
    paramInfo_.yDtype = yDtype;
    paramInfo_.yFormat = yFormat;

    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    auto permPtr = attrs->GetListInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, permPtr);
    paramInfo_.perm.SetDimNum(permPtr->GetSize());
    for (int64_t i = 0; i < int64_t(paramInfo_.perm.GetDimNum()); i++) {
        paramInfo_.perm[i] = permPtr->GetData()[i];
    }

    auto shapePtr = attrs->GetListInt(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, shapePtr);
    paramInfo_.shape.SetDimNum(shapePtr->GetSize());
    for (int64_t i = 0; i < int64_t(paramInfo_.shape.GetDimNum()); i++) {
        paramInfo_.shape[i] = shapePtr->GetData()[i];
    }

    auto transposeFirstPtr = attrs->GetAttrPointer<bool>(2);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, transposeFirstPtr);
    paramInfo_.transpose_first = *transposeFirstPtr;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionTransposeDTiling::ParametersVerifyingFormatAndDatatype()
{
    if (paramInfo_.xFormat != ge::FORMAT_FRACTAL_NZ) {
        paramInfo_.xFormat = ge::FORMAT_ND;
    }
    if (paramInfo_.xFormat == ge::FORMAT_ND) {
        OP_CHECK_IF(
            std::find(SUPPORT_DTYPE_ND.begin(), SUPPORT_DTYPE_ND.end(), paramInfo_.xDtype) == SUPPORT_DTYPE_ND.end(),
            OP_LOGE_FOR_INVALID_DTYPE(
                tilingContext_->GetNodeName(), "x", Ops::Base::ToString(paramInfo_.xDtype).c_str(),
                "INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT16, FLOAT and BF16"),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(
            tilingContext_->GetNodeName(), "x", Ops::Base::ToString(paramInfo_.xFormat).c_str(), "ND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionTransposeDTiling::ParametersVerifyingInputAndOutput()
{
    if (paramInfo_.xDtype != paramInfo_.yDtype) {
        std::string dtypeMsg =
            Ops::Base::ToString(paramInfo_.xDtype) + " and " + Ops::Base::ToString(paramInfo_.yDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            tilingContext_->GetNodeName(), "x and y", dtypeMsg.c_str(), "The dtypes of x and y must be the same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionTransposeDTiling::ParametersVerifyingDimNd()
{
    if (paramInfo_.xShape.GetDimNum() > MAX_DIM_ND || paramInfo_.xShape.GetDimNum() < MIN_DIM_ND) {
        std::string dimExpectRange = "between " + std::to_string(MIN_DIM_ND) + " and " + std::to_string(MAX_DIM_ND);
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            tilingContext_->GetNodeName(), "x", std::to_string(paramInfo_.xShape.GetDimNum()).c_str(),
            dimExpectRange.c_str());
        return ge::GRAPH_FAILED;
    }

    if (paramInfo_.yShape.GetDimNum() > MAX_DIM_ND || paramInfo_.yShape.GetDimNum() < MIN_DIM_ND) {
        std::string dimExpectRange = "between " + std::to_string(MIN_DIM_ND) + " and " + std::to_string(MAX_DIM_ND);
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            tilingContext_->GetNodeName(), "y", std::to_string(paramInfo_.yShape.GetDimNum()).c_str(),
            dimExpectRange.c_str());
        return ge::GRAPH_FAILED;
    }

    if (paramInfo_.shape.GetDimNum() > MAX_DIM_ND || paramInfo_.shape.GetDimNum() < MIN_DIM_ND) {
        std::string dimExpectRange = "between " + std::to_string(MIN_DIM_ND) + " and " + std::to_string(MAX_DIM_ND);
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            tilingContext_->GetNodeName(), "shape", std::to_string(paramInfo_.shape.GetDimNum()).c_str(),
            dimExpectRange.c_str());
        return ge::GRAPH_FAILED;
    }

    if (paramInfo_.transpose_first) {
        if (paramInfo_.perm.GetDimNum() != paramInfo_.xShape.GetDimNum()) {
            std::string dimMsg =
                std::to_string(paramInfo_.xShape.GetDimNum()) + " and " + std::to_string(paramInfo_.perm.GetDimNum());
            std::string reasonMsg = "The shape dimensions of perm and x must be the same, when input x need to be transposed first";
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                tilingContext_->GetNodeName(), "x and perm", dimMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (paramInfo_.perm.GetDimNum() != paramInfo_.shape.GetDimNum()) {
            std::string dimMsg =
                std::to_string(paramInfo_.perm.GetDimNum()) + " and " + std::to_string(paramInfo_.shape.GetDimNum());
            std::string reasonMsg =
                "The shape dimensions of parameter perm and parameter shape must be the same, when input x does not need "
                "to be transposed first";
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                tilingContext_->GetNodeName(), "perm and shape", dimMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionTransposeDTiling::ParametersVerifyingDimNz()
{
    if (paramInfo_.xShape.GetDimNum() > MAX_DIM_NZ || paramInfo_.xShape.GetDimNum() < MIN_DIM_NZ) {
        std::string dimExpectRange = "between " + std::to_string(MIN_DIM_NZ) + " and " + std::to_string(MAX_DIM_NZ);
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            tilingContext_->GetNodeName(), "x", std::to_string(paramInfo_.xShape.GetDimNum()).c_str(),
            dimExpectRange.c_str());
        return ge::GRAPH_FAILED;
    }

    if (paramInfo_.yShape.GetDimNum() > MAX_DIM_NZ || paramInfo_.yShape.GetDimNum() < MIN_DIM_NZ) {
        std::string dimExpectRange = "between " + std::to_string(MIN_DIM_NZ) + " and " + std::to_string(MAX_DIM_NZ);
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            tilingContext_->GetNodeName(), "y", std::to_string(paramInfo_.yShape.GetDimNum()).c_str(),
            dimExpectRange.c_str());
        return ge::GRAPH_FAILED;
    }

    if (paramInfo_.shape.GetDimNum() > MAX_DIM_NZ2ND || paramInfo_.shape.GetDimNum() < MIN_DIM_NZ2ND) {
        std::string dimExpectRange = "between " + std::to_string(MIN_DIM_NZ2ND) + " and " + std::to_string(MAX_DIM_NZ2ND);
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            tilingContext_->GetNodeName(), "shape", std::to_string(paramInfo_.shape.GetDimNum()).c_str(),
            dimExpectRange.c_str());
        return ge::GRAPH_FAILED;
    }

    if (paramInfo_.transpose_first) {
        if (paramInfo_.perm.GetDimNum() != paramInfo_.xShape.GetDimNum() - 2) {
            std::string dimMsg =
                std::to_string(paramInfo_.xShape.GetDimNum()) + " and " + std::to_string(paramInfo_.perm.GetDimNum());
            std::string reasonMsg = "The shape dimension of perm must be twice that of x, when format is NZ";
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                tilingContext_->GetNodeName(), "x and perm", dimMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (paramInfo_.perm.GetDimNum() != paramInfo_.shape.GetDimNum()) {
            std::string dimMsg =
                std::to_string(paramInfo_.perm.GetDimNum()) + " and " + std::to_string(paramInfo_.shape.GetDimNum());
            std::string reasonMsg = "The shape dimensions of parameter perm and parameter shape must be the same";
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                tilingContext_->GetNodeName(), "perm and shape", dimMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionTransposeDTiling::ParametersVerifyingPerm()
{
    int64_t checkPerm = 0;
    for (int64_t i = 0; i < int64_t(paramInfo_.perm.GetDimNum()); i++) {
        if (paramInfo_.perm[i] < int64_t(paramInfo_.perm.GetDimNum())) {
            checkPerm += (1l << paramInfo_.perm[i]);
        }
    }
    if (checkPerm != (1l << paramInfo_.perm.GetDimNum()) - 1) {
        OP_LOGE_FOR_INVALID_VALUE(
            tilingContext_->GetNodeName(), "perm", Ops::Base::ToString(paramInfo_.perm).c_str(),
            "a permutation of 0,1, ..., dim - 1");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionTransposeDTiling::ParametersVerifyingProdAndPositive()
{
    int64_t prodX = 1;
    int64_t prodY = 1;
    int64_t prodShape = 1;
    bool xPositive = true;
    bool yPositive = true;
    bool shapePositive = true;
    for (int64_t i = 0; i < int64_t(paramInfo_.xShape.GetDimNum()); i++) {
        xPositive &= (paramInfo_.xShape.GetDim(i) > 0);
        prodX *= paramInfo_.xShape.GetDim(i);
    }
    for (int64_t i = 0; i < int64_t(paramInfo_.yShape.GetDimNum()); i++) {
        yPositive &= (paramInfo_.yShape.GetDim(i) > 0);
        prodY *= paramInfo_.yShape.GetDim(i);
    }
    for (int64_t i = 0; i < int64_t(paramInfo_.shape.GetDimNum()); i++) {
        shapePositive &= (paramInfo_.shape.GetDim(i) > 0);
        prodShape *= paramInfo_.shape.GetDim(i);
    }

    OP_CHECK_IF(
        xPositive == false,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            tilingContext_->GetNodeName(), "x", Ops::Base::ToString(paramInfo_.xShape).c_str(),
            "All axes of x must be positive numbers"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        yPositive == false,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            tilingContext_->GetNodeName(), "y", Ops::Base::ToString(paramInfo_.yShape).c_str(),
            "All axes of output y must be positive numbers"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        shapePositive == false,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            tilingContext_->GetNodeName(), "shape", Ops::Base::ToString(paramInfo_.shape).c_str(),
            "All axes of parameter shape must be positive numbers"),
        return ge::GRAPH_FAILED);

    if (prodX > prodY || prodX != prodShape) {
        std::string shapeMsg = std::to_string(prodX) + ", " + std::to_string(prodY) + " and " + std::to_string(prodShape);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            tilingContext_->GetNodeName(), "x, y and shape", shapeMsg.c_str(),
            "The shape sizes of parameter x and parameter shape must be the same, and the shape size of x must less "
            "than or equal to that of y");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionTransposeDTiling::ParametersVerifying()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start ConfusionTransposeDTiling ParametersVerifying.");

    if (GetParameters() == ge::GRAPH_FAILED || ParametersVerifyingFormatAndDatatype() == ge::GRAPH_FAILED ||
        ParametersVerifyingInputAndOutput() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (paramInfo_.xFormat == ge::FORMAT_ND && ParametersVerifyingDimNd() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (paramInfo_.xFormat == ge::FORMAT_FRACTAL_NZ && ParametersVerifyingDimNz() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (ParametersVerifyingPerm() == ge::GRAPH_FAILED || ParametersVerifyingProdAndPositive() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(
        tilingContext_->GetNodeName(),
        "x shape is %s, output shape is %s, shape is %s, perm is %s, transpose_first is %d",
        Ops::Base::ToString(paramInfo_.xShape).c_str(), Ops::Base::ToString(paramInfo_.yShape).c_str(),
        Ops::Base::ToString(paramInfo_.shape).c_str(), Ops::Base::ToString(paramInfo_.perm).c_str(), paramInfo_.transpose_first);
    return ge::GRAPH_SUCCESS;
}

void ConfusionTransposeDTiling::ProcessShapeInfoForNz(ShapeInfo& shapeInfo)
{
    gert::Shape xNDShape;
    int64_t dimNumX = paramInfo_.xShape.GetDimNum();
    for (int64_t i = 0; i < dimNumX - DIM_FOUR; i++) {
        xNDShape.SetDim(i, paramInfo_.xShape[i]);
    }
    xNDShape.SetDim(dimNumX - DIM_FOUR, paramInfo_.xShape[dimNumX - DIM_THREE] * paramInfo_.xShape[dimNumX - DIM_TWO]);
    xNDShape.SetDim(dimNumX - DIM_THREE, paramInfo_.xShape[dimNumX - DIM_FOUR] * paramInfo_.xShape[dimNumX - DIM_ONE]);
    xNDShape.SetDimNum(dimNumX - DIM_TWO);

    vector<int64_t> fracNzIn;
    vector<int64_t> finalPerm;
    if (paramInfo_.transpose_first) {
        vector<int64_t> xNDShapeVec = TransShapeToVector(xNDShape);
        vector<int64_t> permVec = TransShapeToVector(paramInfo_.perm);
        vector<int64_t> shapeVec = TransShapeToVector(paramInfo_.shape);
        vector<int64_t> reshapeIn = ShapeAfterTranspose(xNDShapeVec, permVec);

        TransposeReshape(permVec, reshapeIn, shapeVec, fracNzIn, finalPerm);
    } else {
        vector<int64_t> reshapeIn = TransShapeToVector(xNDShape);
        vector<int64_t> permVec = TransShapeToVector(paramInfo_.perm);
        vector<int64_t> shapeVec = TransShapeToVector(paramInfo_.shape);

        ReshapeTranspose(permVec, reshapeIn, shapeVec, fracNzIn, finalPerm);
    }
    if (fracNzIn.size() > MAX_DIM_NZ || fracNzIn.size() < MIN_DIM_NZ || finalPerm.size() > MAX_DIM_NZ ||
        finalPerm.size() < MIN_DIM_NZ) {
        std::string msg = "Invalid fracNzIn or finalPerm's dim, which should between 1 and 8. Error shape is" +
                     VectorToString(fracNzIn) + " and " + VectorToString(finalPerm);
        throw std::invalid_argument(msg.c_str());
    }
    OP_LOGD(
        "ConfusionTransposeD", "fracNzIn: %s, finalPerm %s", VectorToString(fracNzIn).c_str(),
        VectorToString(finalPerm).c_str());
    shapeInfo.inShapeSize = fracNzIn.size();
    shapeInfo.outShapeSize = fracNzIn.size();
    shapeInfo.permSize = finalPerm.size();
    shapeInfo.dim = fracNzIn.size();
    shapeInfo.origDim = fracNzIn.size();
    for (int64_t i = 0; i < shapeInfo.inShapeSize; i++) {
        shapeInfo.inShape[i] = fracNzIn[i];
        shapeInfo.outShape[i] = fracNzIn[finalPerm[i]];
        shapeInfo.perm[i] = finalPerm[i];
    }
}

void ConfusionTransposeDTiling::ProcessShapeInfoForNd(ShapeInfo& shapeInfo)
{
    if (paramInfo_.transpose_first) {
        shapeInfo.inShapeSize = paramInfo_.xShape.GetDimNum();
        shapeInfo.outShapeSize = paramInfo_.xShape.GetDimNum();
        for (int64_t i = 0; i < shapeInfo.inShapeSize; i++) {
            shapeInfo.inShape[i] = paramInfo_.xShape[i];
            shapeInfo.outShape[i] = paramInfo_.xShape[paramInfo_.perm[i]];
        }
        shapeInfo.dim = paramInfo_.xShape.GetDimNum();
        shapeInfo.origDim = paramInfo_.xShape.GetDimNum();
    } else {
        shapeInfo.inShapeSize = paramInfo_.shape.GetDimNum();
        shapeInfo.outShapeSize = paramInfo_.shape.GetDimNum();
        for (int64_t i = 0; i < shapeInfo.inShapeSize; i++) {
            shapeInfo.inShape[i] = paramInfo_.shape[i];
            shapeInfo.outShape[i] = paramInfo_.shape[paramInfo_.perm[i]];
        }
        shapeInfo.dim = paramInfo_.shape.GetDimNum();
        shapeInfo.origDim = paramInfo_.shape.GetDimNum();
    }
    shapeInfo.permSize = paramInfo_.perm.GetDimNum();
    for (int64_t i = 0; i < shapeInfo.permSize; i++) {
        shapeInfo.perm[i] = paramInfo_.perm[i];
    }
}

void ConfusionTransposeDTiling::ProcessShapeInfo(ShapeInfo& shapeInfo)
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start ConfusionTransposeDTiling ProcessShapeInfo.");
    shapeInfo.eleLenInBytes = ge::GetSizeByDataType(paramInfo_.xDtype);

    if (paramInfo_.xFormat == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGD(tilingContext_->GetNodeName(), "Is Nz tiling.");
        ProcessShapeInfoForNz(shapeInfo);
    } else {
        OP_LOGD(tilingContext_->GetNodeName(), "Is Nd tiling.");
        ProcessShapeInfoForNd(shapeInfo);
    }
}

static ge::graphStatus DslTilingForRelatedToTranspose(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Dsl Tiling is running.");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionTransposeDTilingForAscendC(gert::TilingContext* context, uint64_t coreNum, uint64_t ubSize)
{
    OP_LOGD(context->GetNodeName(), "Start ConfusionTransposeDTilingForAscendC.");
    ConfusionTransposeDTiling tilingObject(context);

    OP_CHECK_IF(
        tilingObject.ParametersVerifying() != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "ConfusionTransposeDTiling failed to verify params!"),
        return ge::GRAPH_FAILED);

    // construct an equivalent Transpose inputShapeInfo
    ShapeInfo inputShapeInfo;
    try {
        tilingObject.ProcessShapeInfo(inputShapeInfo);
    } catch (const std::invalid_argument& e) {
        OP_LOGE(context->GetNodeName(), "Failed to set shape info, reason: %s", e.what());
        return ge::GRAPH_FAILED;
    }

    ConfusionTransposeDTilingData tilingData;
    ConfusionTransposeDCompileInfo compileInfo;
    compileInfo.transposeCompilerInfo.coreNum = coreNum;
    compileInfo.transposeCompilerInfo.ubSize = ubSize;

    TransposeNddmaTiling transposeTilingObject(context);
    OP_CHECK_IF(
        (transposeTilingObject.TilingForReleatedTranspose(
             context, &tilingData.transposeOpTiling, &compileInfo.transposeCompilerInfo, inputShapeInfo) ==
         ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(), "Transpose Tiling failed"), return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    OP_LOGD(context->GetNodeName(), "ConfusionTransposeDTilingForAscendC success.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForConfusionTransposeD(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to do TilingForConfusionTransposeD");

    OP_LOGD(context->GetNodeName(), "in ascendc");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (coreNum <= 0),
        OP_LOGE(context->GetNodeName(), "ConfusionTransposeD Op GetHardwareInfo Failed, coreNum: %lu.", coreNum),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        (ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "ConfusionTransposeD Op GetHardwareInfo Failed, ubSize: %lu.", ubSize),
        return ge::GRAPH_FAILED);

    return ConfusionTransposeDTilingForAscendC(context, coreNum, ubSize);
    OP_LOGD(context->GetNodeName(), "in dsl");
    return DslTilingForRelatedToTranspose(context);
}

static ge::graphStatus TilingPrepareConfusionTransposeDForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start TilingPrepareConfusionTransposeDForAscendC");
    auto ci = context->GetCompiledInfo<TransposeCompilerInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (ci->coreNum <= 0),
        OP_LOGE(
            context->GetNodeName(), "ConfusionTransposeD Op GetHardwareInfo Failed, coreNum:%ld.", ci->coreNum),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (ci->ubSize <= 0),
        OP_LOGE(
            context->GetNodeName(), "ConfusionTransposeD Op GetHardwareInfo Failed, ubSize:%ld.", ci->ubSize),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "ConfusionTransposeD Op get coreNum:%lu, ubSize:%lu.", ci->coreNum, ci->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForConfusionTransposeD(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepareForConfusionTransposeD.");
    auto ci = context->GetCompiledInfo<TransposeCompilerInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    TilingPrepareConfusionTransposeDForAscendC(context);   
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ConfusionTransposeD)
    .Tiling(TilingForConfusionTransposeD)
    .TilingParse<ConfusionTransposeDCompileInfo>(TilingPrepareForConfusionTransposeD);

} // namespace optiling
