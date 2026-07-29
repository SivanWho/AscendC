#include "aclnn/acl_meta.h"

#include <cstdint>

extern "C" aclnnStatus aclnnSquareSumV1GeneratedGetWorkspaceSize(
    const aclTensor *x,
    const aclIntArray *axis,
    bool keepDims,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern "C" __attribute__((visibility("default"))) aclnnStatus
aclnnSquareSumV1GetWorkspaceSize(
    const aclTensor *x,
    const aclIntArray *axis,
    bool keepDims,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    if (x == nullptr || axis == nullptr || out == nullptr ||
        workspaceSize == nullptr || executor == nullptr) {
        return 161001;
    }

    uint64_t axisCount = 0;
    aclnnStatus status = aclGetIntArraySize(axis, &axisCount);
    if (status != 0 || axisCount != 0) {
        return status == 0
            ? aclnnSquareSumV1GeneratedGetWorkspaceSize(
                  x, axis, keepDims, out, workspaceSize, executor)
            : status;
    }

    int64_t *shape = nullptr;
    uint64_t rank = 0;
    status = aclGetViewShape(x, &shape, &rank);
    if (status != 0) {
        return status;
    }
    if (rank > 8) {
        return 161002;
    }

    int64_t normalizedAxes[8] = {};
    uint64_t normalizedCount = rank;
    for (uint64_t i = 0; i < rank; ++i) {
        normalizedAxes[i] = static_cast<int64_t>(i);
    }
    if (rank == 0) {
        normalizedAxes[0] = 0;
        normalizedCount = 1;
    }

    aclIntArray *normalized =
        aclCreateIntArray(normalizedAxes, normalizedCount);
    if (normalized == nullptr) {
        return 161001;
    }
    status = aclnnSquareSumV1GeneratedGetWorkspaceSize(
        x, normalized, keepDims, out, workspaceSize, executor);
    aclDestroyIntArray(normalized);
    return status;
}
