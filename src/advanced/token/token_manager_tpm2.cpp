#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/common/logger.h"
#include <cstring>

#if defined(VGRE_HAS_TPM2)
#include <tss2/tss2_esys.h>
#include <tss2/tss2_mu.h>
#include <tss2/tss2_tctildr.h>

namespace vgre {
namespace advanced {

VGREResult HardwareTokenManager::initTPM() {
    TSS2_RC rc;
    TSS2_TCTI_CONTEXT* tcti_ctx = nullptr;
    
    rc = Tss2_TctiLdr_Initialize("device:/dev/tpmrm0", &tcti_ctx);
    if (rc != TSS2_RC_SUCCESS) {
        rc = Tss2_TctiLdr_Initialize("device:/dev/tpm0", &tcti_ctx);
        if (rc != TSS2_RC_SUCCESS) {
            VGRE_LOG_DEBUG("HardwareTokenManager", "Physical TPM device not available (rc=" + std::to_string(rc) + ")");
            return VGREResult::ERR_NOT_SUPPORTED;
        }
    }
    
    rc = Esys_Initialize(&tpm_context_, tcti_ctx, NULL);
    if (rc != TSS2_RC_SUCCESS) {
        VGRE_LOG_ERROR("HardwareTokenManager", "Failed to initialize TPM ESYS context (rc=" + std::to_string(rc) + ")");
        Tss2_TctiLdr_Finalize(&tcti_ctx);
        return VGREResult::ERR_NOT_SUPPORTED;
    }
    
    tpm_nv_handle_ = 0x01C00100;
    
    VGRE_LOG_INFO("HardwareTokenManager", "TPM 2.0 initialized successfully");
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::storeTPM(const std::string& service, const std::string& token) {
    if (!tpm_context_) {
        return VGREResult::ERR_NOT_INITIALIZED;
    }
    
    TSS2_RC rc;
    
    uint32_t service_hash = 0;
    for (char c : service) {
        service_hash = service_hash * 31 + static_cast<uint32_t>(c);
    }
    TPM2_HANDLE nv_index = tpm_nv_handle_ + (service_hash % 0x100);
    
    ESYS_TR nv_handle = ESYS_TR_NONE;
    rc = Esys_TR_FromTPMPublic(tpm_context_, nv_index, 
                               ESYS_TR_NONE, ESYS_TR_NONE, ESYS_TR_NONE,
                               &nv_handle);
    
    if (rc == TSS2_RC_SUCCESS) {
        rc = Esys_NV_UndefineSpace(tpm_context_,
                                   ESYS_TR_RH_OWNER,
                                   nv_handle,
                                   ESYS_TR_PASSWORD, ESYS_TR_NONE, ESYS_TR_NONE);
        if (rc != TSS2_RC_SUCCESS) {
            VGRE_LOG_ERROR("HardwareTokenManager", "Failed to undefine existing NV space (rc=" + std::to_string(rc) + ")");
            return VGREResult::ERR_IO;
        }
    }
    
    TPM2B_AUTH auth = {.size = 0};
    TPM2B_NV_PUBLIC publicInfo = {
        .size = 0,
        .nvPublic = {
            .nvIndex = nv_index,
            .nameAlg = TPM2_ALG_SHA256,
            .attributes = TPMA_NV_AUTHREAD | TPMA_NV_AUTHWRITE | 
                         TPMA_NV_OWNERWRITE | TPMA_NV_OWNERREAD,
            .authPolicy = {.size = 0},
            .dataSize = static_cast<uint16_t>(token.size())
        }
    };
    
    rc = Esys_NV_DefineSpace(tpm_context_,
                            ESYS_TR_RH_OWNER,
                            ESYS_TR_PASSWORD, ESYS_TR_NONE, ESYS_TR_NONE,
                            &auth,
                            &publicInfo,
                            &nv_handle);
    
    if (rc != TSS2_RC_SUCCESS) {
        VGRE_LOG_ERROR("HardwareTokenManager", "Failed to define NV space (rc=" + std::to_string(rc) + ")");
        return VGREResult::ERR_IO;
    }
    
    TPM2B_MAX_NV_BUFFER nv_write_data;
    nv_write_data.size = token.size();
    std::memcpy(nv_write_data.buffer, token.c_str(), token.size());
    
    rc = Esys_NV_Write(tpm_context_,
                      nv_handle,
                      nv_handle,
                      ESYS_TR_PASSWORD, ESYS_TR_NONE, ESYS_TR_NONE,
                      &nv_write_data,
                      0);
    
    if (rc != TSS2_RC_SUCCESS) {
        VGRE_LOG_ERROR("HardwareTokenManager", "Failed to write to NV space (rc=" + std::to_string(rc) + ")");
        return VGREResult::ERR_IO;
    }
    
    VGRE_LOG_INFO("HardwareTokenManager", "Token stored in TPM NV index 0x" + std::to_string(nv_index));
    
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::getTPM(const std::string& service, std::string& outToken) {
    if (!tpm_context_) {
        return VGREResult::ERR_NOT_INITIALIZED;
    }
    
    TSS2_RC rc;
    
    uint32_t service_hash = 0;
    for (char c : service) {
        service_hash = service_hash * 31 + static_cast<uint32_t>(c);
    }
    TPM2_HANDLE nv_index = tpm_nv_handle_ + (service_hash % 0x100);
    
    ESYS_TR nv_handle = ESYS_TR_NONE;
    rc = Esys_TR_FromTPMPublic(tpm_context_, nv_index,
                               ESYS_TR_NONE, ESYS_TR_NONE, ESYS_TR_NONE,
                               &nv_handle);
    
    if (rc != TSS2_RC_SUCCESS) {
        return VGREResult::ERR_AUTH_FAILED;
    }
    
    TPM2B_NV_PUBLIC* nv_public = nullptr;
    TPM2B_NAME* nv_name = nullptr;
    rc = Esys_NV_ReadPublic(tpm_context_,
                           nv_handle,
                           ESYS_TR_NONE, ESYS_TR_NONE, ESYS_TR_NONE,
                           &nv_public,
                           &nv_name);
    
    if (rc != TSS2_RC_SUCCESS) {
        return VGREResult::ERR_IO;
    }
    
    uint16_t data_size = nv_public->nvPublic.dataSize;
    Esys_Free(nv_public);
    Esys_Free(nv_name);
    
    TPM2B_MAX_NV_BUFFER* nv_data = nullptr;
    rc = Esys_NV_Read(tpm_context_,
                     nv_handle,
                     nv_handle,
                     ESYS_TR_PASSWORD, ESYS_TR_NONE, ESYS_TR_NONE,
                     data_size,
                     0,
                     &nv_data);
    
    if (rc != TSS2_RC_SUCCESS) {
        VGRE_LOG_ERROR("HardwareTokenManager", "Failed to read from NV space (rc=" + std::to_string(rc) + ")");
        return VGREResult::ERR_IO;
    }
    
    outToken = std::string(reinterpret_cast<const char*>(nv_data->buffer), nv_data->size);
    Esys_Free(nv_data);
    
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::deleteTPM(const std::string& service) {
    if (!tpm_context_) {
        return VGREResult::ERR_NOT_INITIALIZED;
    }
    
    TSS2_RC rc;
    
    uint32_t service_hash = 0;
    for (char c : service) {
        service_hash = service_hash * 31 + static_cast<uint32_t>(c);
    }
    TPM2_HANDLE nv_index = tpm_nv_handle_ + (service_hash % 0x100);
    
    ESYS_TR nv_handle = ESYS_TR_NONE;
    rc = Esys_TR_FromTPMPublic(tpm_context_, nv_index,
                               ESYS_TR_NONE, ESYS_TR_NONE, ESYS_TR_NONE,
                               &nv_handle);
    
    if (rc != TSS2_RC_SUCCESS) {
        return VGREResult::ERR_AUTH_FAILED;
    }
    
    rc = Esys_NV_UndefineSpace(tpm_context_,
                               ESYS_TR_RH_OWNER,
                               nv_handle,
                               ESYS_TR_PASSWORD, ESYS_TR_NONE, ESYS_TR_NONE);
    
    if (rc != TSS2_RC_SUCCESS) {
        VGRE_LOG_ERROR("HardwareTokenManager", "Failed to undefine NV space (rc=" + std::to_string(rc) + ")");
        return VGREResult::ERR_IO;
    }
    
    return VGREResult::SUCCESS;
}

} // namespace advanced
} // namespace vgre

#else

namespace vgre {
namespace advanced {
VGREResult HardwareTokenManager::initTPM() { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::storeTPM(const std::string&, const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::getTPM(const std::string&, std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::deleteTPM(const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
} // namespace advanced
} // namespace vgre

#endif
