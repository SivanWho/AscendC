#include "aclnn/acl_meta.h"

extern "C" aclnnStatus aclnnGreaterCustomGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *other,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern "C" aclnnStatus aclnnGreaterCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

extern "C" __attribute__((visibility("default"))) aclnnStatus aclnnGreaterGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *other,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnGreaterCustomGetWorkspaceSize(self, other, out, workspaceSize, executor);
}

extern "C" __attribute__((visibility("default"))) aclnnStatus aclnnGreater(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return aclnnGreaterCustom(workspace, workspaceSize, executor, stream);
}
