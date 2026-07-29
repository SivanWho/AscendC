#include "aclnn/acl_meta.h"

extern "C" aclnnStatus aclnnIndexAddCustomGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *index,
    const aclTensor *source,
    int64_t dim,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern "C" __attribute__((visibility("default"))) aclnnStatus aclnnIndexAddGetWorkspaceSize(
    const aclTensor *self,
    int64_t dim,
    const aclTensor *index,
    const aclTensor *source,
    const aclScalar *alpha,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    if (alpha == nullptr) {
        return 161001;
    }
    return aclnnIndexAddCustomGetWorkspaceSize(
        self, index, source, dim, out, workspaceSize, executor);
}
