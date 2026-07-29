#include "aclnn/acl_meta.h"

extern "C" aclnnStatus aclnnConcatCustomGetWorkspaceSize(
    const aclTensorList *inputs,
    int64_t dim,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern "C" aclnnStatus aclnnConcatCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

extern "C" __attribute__((visibility("default"))) aclnnStatus aclnnConcatGetWorkspaceSize(
    const aclTensorList *inputs,
    int64_t dim,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnConcatCustomGetWorkspaceSize(inputs, dim, out, workspaceSize, executor);
}

extern "C" __attribute__((visibility("default"))) aclnnStatus aclnnConcat(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return aclnnConcatCustom(workspace, workspaceSize, executor, stream);
}
