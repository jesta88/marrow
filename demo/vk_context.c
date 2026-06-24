#include "vk_context.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "png_write.h"
#include "profiler.h"   /* prof_now_s - the demo's single monotonic timer */

#define VKCHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "vk error %d at %s:%d\n", (int)_r, __FILE__, __LINE__); return 0; } } while (0)

static void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                          VkImageLayout old_layout, VkImageLayout new_layout,
                          VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                          VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access);

/* ------------------------------------------------------------------ debug messenger */

static VKAPI_ATTR VkBool32 VKAPI_CALL
debug_cb(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT type,
         const VkDebugUtilsMessengerCallbackDataEXT *data, void *user) {
    (void)type; (void)user;
    if (sev & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT))
        fprintf(stderr, "[vk] %s\n", data->pMessage);
    return VK_FALSE;
}

static int has_layer(const char *name) {
    uint32_t n = 0; vkEnumerateInstanceLayerProperties(&n, NULL);
    VkLayerProperties *props = calloc(n, sizeof *props);
    vkEnumerateInstanceLayerProperties(&n, props);
    int found = 0;
    for (uint32_t i = 0; i < n; ++i) if (strcmp(props[i].layerName, name) == 0) { found = 1; break; }
    free(props);
    return found;
}

/* ------------------------------------------------------------------ instance */

static int create_instance(VkCtx *c, int want_validation) {
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "marrow-demo";
    app.apiVersion = VK_API_VERSION_1_4;

    uint32_t glfw_ext_count = 0;
    const char **glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

    const char *exts[16]; uint32_t ext_count = 0;
    for (uint32_t i = 0; i < glfw_ext_count; ++i) exts[ext_count++] = glfw_exts[i];

    const char *layers[2]; uint32_t layer_count = 0;
    int validation = want_validation && has_layer("VK_LAYER_KHRONOS_validation");
    if (validation) {
        layers[layer_count++] = "VK_LAYER_KHRONOS_validation";
        exts[ext_count++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    }

    VkInstanceCreateInfo ci = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = ext_count;
    ci.ppEnabledExtensionNames = exts;
    ci.enabledLayerCount = layer_count;
    ci.ppEnabledLayerNames = layers;
    VKCHECK(vkCreateInstance(&ci, NULL, &c->instance));
    volkLoadInstance(c->instance);

    if (validation) {
        VkDebugUtilsMessengerCreateInfoEXT di = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
        di.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        di.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        di.pfnUserCallback = debug_cb;
        vkCreateDebugUtilsMessengerEXT(c->instance, &di, NULL, &c->debug);
    }
    return 1;
}

/* ------------------------------------------------------------------ physical device + features */

static int device_has_ext(VkPhysicalDevice pd, const char *name) {
    uint32_t n = 0; vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL);
    VkExtensionProperties *props = calloc(n, sizeof *props);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &n, props);
    int found = 0;
    for (uint32_t i = 0; i < n; ++i) if (strcmp(props[i].extensionName, name) == 0) { found = 1; break; }
    free(props);
    return found;
}

/* Find a queue family with graphics + present. Returns UINT32_MAX if none. On success writes the
 * family's timestampValidBits to *out_valid_bits (0 => the queue can't time-stamp). */
static uint32_t pick_queue_family(VkCtx *c, VkPhysicalDevice pd, uint32_t *out_valid_bits) {
    uint32_t n = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, NULL);
    VkQueueFamilyProperties *qf = calloc(n, sizeof *qf);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, qf);
    uint32_t result = UINT32_MAX;
    for (uint32_t i = 0; i < n; ++i) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, c->surface, &present);
        if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
            result = i;
            *out_valid_bits = qf[i].timestampValidBits;
            break;
        }
    }
    free(qf);
    return result;
}

static int device_meets_requirements(VkPhysicalDevice pd) {
    VkPhysicalDeviceShaderObjectFeaturesEXT so = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT };
    VkPhysicalDeviceVulkan13Features v13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    VkPhysicalDeviceVulkan12Features v12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    v13.pNext = &so;
    v12.pNext = &v13;
    VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    f2.pNext = &v12;
    vkGetPhysicalDeviceFeatures2(pd, &f2);

    if (!device_has_ext(pd, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) return 0;
    if (!device_has_ext(pd, VK_EXT_SHADER_OBJECT_EXTENSION_NAME)) return 0;
    if (!so.shaderObject) return 0;
    if (!v13.dynamicRendering || !v13.synchronization2) return 0;
    if (!v12.timelineSemaphore || !v12.bufferDeviceAddress) return 0;
    if (!v12.runtimeDescriptorArray || !v12.descriptorBindingPartiallyBound) return 0;
    if (!v12.shaderSampledImageArrayNonUniformIndexing) return 0;
    if (!v12.descriptorBindingSampledImageUpdateAfterBind) return 0;
    /* also probe every other feature create_device() turns on, or vkCreateDevice fails late with an
     * opaque error on a device that lacks one. scalarBlockLayout underpins the whole CPU<->GPU ABI. */
    if (!v12.scalarBlockLayout) return 0;
    if (!v12.descriptorBindingVariableDescriptorCount) return 0;
    if (!v12.descriptorBindingUpdateUnusedWhilePending) return 0;
    return 1;
}

static int pick_physical_device(VkCtx *c) {
    uint32_t n = 0; vkEnumeratePhysicalDevices(c->instance, &n, NULL);
    if (n == 0) { fprintf(stderr, "no Vulkan devices\n"); return 0; }
    VkPhysicalDevice *devs = calloc(n, sizeof *devs);
    vkEnumeratePhysicalDevices(c->instance, &n, devs);

    VkPhysicalDevice chosen = VK_NULL_HANDLE; uint32_t chosen_qf = UINT32_MAX; int chosen_discrete = 0;
    uint32_t chosen_valid_bits = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (!device_meets_requirements(devs[i])) continue;
        uint32_t valid_bits = 0;
        uint32_t qf = pick_queue_family(c, devs[i], &valid_bits);
        if (qf == UINT32_MAX) continue;
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(devs[i], &props);
        int discrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        if (chosen == VK_NULL_HANDLE || (discrete && !chosen_discrete)) {
            chosen = devs[i]; chosen_qf = qf; chosen_discrete = discrete; chosen_valid_bits = valid_bits;
        }
    }
    free(devs);
    if (chosen == VK_NULL_HANDLE) {
        fprintf(stderr, "no device meets the modern feature requirements "
                        "(shader objects / dynamic rendering / descriptor indexing / BDA)\n");
        return 0;
    }
    c->phys = chosen;
    c->queue_family = chosen_qf;
    c->shader_object_ok = 1;
    vkGetPhysicalDeviceMemoryProperties(c->phys, &c->mem_props);
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(c->phys, &props);
    c->timestamp_period_ns = props.limits.timestampPeriod;
    c->timestamp_valid_bits = chosen_valid_bits;   /* 0 => GPU timing disabled (reported "n/a") */
    fprintf(stderr, "[demo] GPU: %s (Vulkan %u.%u.%u), timestamp valid bits %u\n", props.deviceName,
            VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
            VK_API_VERSION_PATCH(props.apiVersion), c->timestamp_valid_bits);
    return 1;
}

static int create_device(VkCtx *c) {
    VkPhysicalDeviceShaderObjectFeaturesEXT so = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT };
    so.shaderObject = VK_TRUE;
    VkPhysicalDeviceVulkan13Features v13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    v13.dynamicRendering = VK_TRUE;
    v13.synchronization2 = VK_TRUE;
    v13.pNext = &so;
    VkPhysicalDeviceVulkan12Features v12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    v12.timelineSemaphore = VK_TRUE;
    v12.bufferDeviceAddress = VK_TRUE;
    v12.runtimeDescriptorArray = VK_TRUE;
    v12.descriptorBindingPartiallyBound = VK_TRUE;
    v12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    v12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    v12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    v12.descriptorBindingVariableDescriptorCount = VK_TRUE;
    v12.scalarBlockLayout = VK_TRUE;
    v12.pNext = &v13;
    VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    f2.pNext = &v12;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qi.queueFamilyIndex = c->queue_family;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;

    const char *dev_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_SHADER_OBJECT_EXTENSION_NAME };
    VkDeviceCreateInfo ci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    ci.pNext = &f2;
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qi;
    ci.enabledExtensionCount = sizeof dev_exts / sizeof dev_exts[0];
    ci.ppEnabledExtensionNames = dev_exts;
    VKCHECK(vkCreateDevice(c->phys, &ci, NULL, &c->device));
    volkLoadDevice(c->device);
    vkGetDeviceQueue(c->device, c->queue_family, 0, &c->queue);
    return 1;
}

/* ------------------------------------------------------------------ swapchain + depth */

static void destroy_swapchain_resources(VkCtx *c) {
    for (uint32_t i = 0; i < c->image_count; ++i) {
        if (c->image_views[i]) vkDestroyImageView(c->device, c->image_views[i], NULL);
        if (c->render_finished[i]) vkDestroySemaphore(c->device, c->render_finished[i], NULL);
        c->image_views[i] = VK_NULL_HANDLE;
        c->render_finished[i] = VK_NULL_HANDLE;
    }
    if (c->depth_view)  vkDestroyImageView(c->device, c->depth_view, NULL);
    if (c->depth_image) vkDestroyImage(c->device, c->depth_image, NULL);
    if (c->depth_mem)   vkFreeMemory(c->device, c->depth_mem, NULL);
    c->depth_view = VK_NULL_HANDLE; c->depth_image = VK_NULL_HANDLE; c->depth_mem = VK_NULL_HANDLE;
}

static int create_depth(VkCtx *c) {
    c->depth_format = VK_FORMAT_D32_SFLOAT;
    VkImageCreateInfo ii = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = c->depth_format;
    ii.extent.width = c->extent.width; ii.extent.height = c->extent.height; ii.extent.depth = 1;
    ii.mipLevels = 1; ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VKCHECK(vkCreateImage(c->device, &ii, NULL, &c->depth_image));

    VkMemoryRequirements mr; vkGetImageMemoryRequirements(c->device, c->depth_image, &mr);
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = vkc_find_memory_type(c, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX) return 0;
    VKCHECK(vkAllocateMemory(c->device, &ai, NULL, &c->depth_mem));
    VKCHECK(vkBindImageMemory(c->device, c->depth_image, c->depth_mem, 0));

    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = c->depth_image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = c->depth_format;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vi.subresourceRange.levelCount = 1; vi.subresourceRange.layerCount = 1;
    VKCHECK(vkCreateImageView(c->device, &vi, NULL, &c->depth_view));
    return 1;
}

static int create_swapchain(VkCtx *c) {
    VkSurfaceCapabilitiesKHR caps;
    VKCHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(c->phys, c->surface, &caps));

    /* extent */
    if (caps.currentExtent.width != UINT32_MAX) {
        c->extent = caps.currentExtent;
    } else {
        int w, h; glfwGetFramebufferSize(c->win, &w, &h);
        c->extent.width = (uint32_t)w; c->extent.height = (uint32_t)h;
    }
    if (c->extent.width == 0 || c->extent.height == 0) return 0; /* minimized */

    /* format: prefer BGRA8 SRGB */
    uint32_t nf = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(c->phys, c->surface, &nf, NULL);
    VkSurfaceFormatKHR *formats = calloc(nf, sizeof *formats);
    vkGetPhysicalDeviceSurfaceFormatsKHR(c->phys, c->surface, &nf, formats);
    VkSurfaceFormatKHR chosen = formats[0];
    for (uint32_t i = 0; i < nf; ++i)
        if (formats[i].format == VK_FORMAT_B8G8R8A8_SRGB &&
            formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { chosen = formats[i]; break; }
    free(formats);
    c->color_format = chosen.format; c->color_space = chosen.colorSpace;

    /* present mode: prefer mailbox, else fifo */
    uint32_t npm = 0; vkGetPhysicalDeviceSurfacePresentModesKHR(c->phys, c->surface, &npm, NULL);
    VkPresentModeKHR *pms = calloc(npm, sizeof *pms);
    vkGetPhysicalDeviceSurfacePresentModesKHR(c->phys, c->surface, &npm, pms);
    c->present_mode = VK_PRESENT_MODE_FIFO_KHR;
    for (uint32_t i = 0; i < npm; ++i) if (pms[i] == VK_PRESENT_MODE_MAILBOX_KHR) { c->present_mode = pms[i]; break; }
    free(pms);

    uint32_t want = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && want > caps.maxImageCount) want = caps.maxImageCount;
    if (want > VKC_MAX_IMAGES) want = VKC_MAX_IMAGES;

    VkSwapchainCreateInfoKHR si = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    si.surface = c->surface;
    si.minImageCount = want;
    si.imageFormat = c->color_format;
    si.imageColorSpace = c->color_space;
    si.imageExtent = c->extent;
    si.imageArrayLayers = 1;
    c->can_capture = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ? 1 : 0;
    si.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    (c->can_capture ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
    si.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    si.preTransform = caps.currentTransform;
    si.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    si.presentMode = c->present_mode;
    si.clipped = VK_TRUE;
    VKCHECK(vkCreateSwapchainKHR(c->device, &si, NULL, &c->swapchain));

    vkGetSwapchainImagesKHR(c->device, c->swapchain, &c->image_count, NULL);
    if (c->image_count > VKC_MAX_IMAGES) c->image_count = VKC_MAX_IMAGES;
    vkGetSwapchainImagesKHR(c->device, c->swapchain, &c->image_count, c->images);

    for (uint32_t i = 0; i < c->image_count; ++i) {
        VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vi.image = c->images[i]; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = c->color_format;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1; vi.subresourceRange.layerCount = 1;
        VKCHECK(vkCreateImageView(c->device, &vi, NULL, &c->image_views[i]));
        VkSemaphoreCreateInfo se = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VKCHECK(vkCreateSemaphore(c->device, &se, NULL, &c->render_finished[i]));
    }
    return create_depth(c);
}

static int recreate_swapchain(VkCtx *c) {
    int w = 0, h = 0;
    glfwGetFramebufferSize(c->win, &w, &h);
    while (w == 0 || h == 0) { glfwWaitEvents(); glfwGetFramebufferSize(c->win, &w, &h); }
    vkDeviceWaitIdle(c->device);
    destroy_swapchain_resources(c);
    if (c->swapchain) vkDestroySwapchainKHR(c->device, c->swapchain, NULL);
    c->swapchain = VK_NULL_HANDLE;
    return create_swapchain(c);
}

static void framebuffer_size_cb(GLFWwindow *win, int w, int h) {
    (void)w; (void)h;
    VkCtx *c = glfwGetWindowUserPointer(win);
    if (c) c->framebuffer_resized = 1;
}

/* ------------------------------------------------------------------ init / destroy */

int vkc_init(VkCtx *c, GLFWwindow *win) {
    memset(c, 0, sizeof *c);
    c->win = win;
    c->clear_rgb[0] = 0.05f; c->clear_rgb[1] = 0.06f; c->clear_rgb[2] = 0.08f;
    glfwSetWindowUserPointer(win, c);
    glfwSetFramebufferSizeCallback(win, framebuffer_size_cb);

#ifdef NDEBUG
    int want_validation = 0;
#else
    int want_validation = 1;
#endif
    if (!create_instance(c, want_validation)) return 0;
    VKCHECK(glfwCreateWindowSurface(c->instance, win, NULL, &c->surface));
    if (!pick_physical_device(c)) return 0;
    if (!create_device(c)) return 0;
    if (!create_swapchain(c)) return 0;

    VkCommandPoolCreateInfo pi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = c->queue_family;
    VKCHECK(vkCreateCommandPool(c->device, &pi, NULL, &c->cmd_pool));

    VkCommandBufferAllocateInfo ci = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ci.commandPool = c->cmd_pool;
    ci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ci.commandBufferCount = VKC_FRAMES_IN_FLIGHT;
    VKCHECK(vkAllocateCommandBuffers(c->device, &ci, c->cmd));

    for (uint32_t i = 0; i < VKC_FRAMES_IN_FLIGHT; ++i) {
        VkSemaphoreCreateInfo se = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VKCHECK(vkCreateSemaphore(c->device, &se, NULL, &c->image_available[i]));
    }
    VkSemaphoreTypeCreateInfo tt = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    tt.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    tt.initialValue = 0;
    VkSemaphoreCreateInfo ti = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    ti.pNext = &tt;
    VKCHECK(vkCreateSemaphore(c->device, &ti, NULL, &c->timeline));
    c->timeline_value = 0;

    /* GPU timestamp query pools - one per frame-in-flight, reset each frame in begin_frame. Skipped
     * when the queue can't time-stamp (the demo then reports GPU stages as "n/a"). */
    if (c->timestamp_valid_bits) {
        VkQueryPoolCreateInfo qpi = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = VKC_GPU_QUERIES_PER_FRAME;
        for (uint32_t f = 0; f < VKC_FRAMES_IN_FLIGHT; ++f)
            VKCHECK(vkCreateQueryPool(c->device, &qpi, NULL, &c->gpu_qpool[f]));
    }
    return 1;
}

void vkc_wait_idle(VkCtx *c) { if (c->device) vkDeviceWaitIdle(c->device); }

void vkc_destroy(VkCtx *c) {
    if (!c->device) return;
    vkDeviceWaitIdle(c->device);
    destroy_swapchain_resources(c);
    if (c->swapchain) vkDestroySwapchainKHR(c->device, c->swapchain, NULL);
    for (uint32_t i = 0; i < VKC_FRAMES_IN_FLIGHT; ++i)
        if (c->image_available[i]) vkDestroySemaphore(c->device, c->image_available[i], NULL);
    for (uint32_t i = 0; i < VKC_FRAMES_IN_FLIGHT; ++i)
        if (c->gpu_qpool[i]) vkDestroyQueryPool(c->device, c->gpu_qpool[i], NULL);
    if (c->timeline) vkDestroySemaphore(c->device, c->timeline, NULL);
    if (c->cmd_pool) vkDestroyCommandPool(c->device, c->cmd_pool, NULL);
    vkDestroyDevice(c->device, NULL);
    if (c->debug) vkDestroyDebugUtilsMessengerEXT(c->instance, c->debug, NULL);
    if (c->surface) vkDestroySurfaceKHR(c->instance, c->surface, NULL);
    vkDestroyInstance(c->instance, NULL);
}

/* ------------------------------------------------------------------ memory / buffers */

uint32_t vkc_find_memory_type(VkCtx *c, uint32_t type_bits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < c->mem_props.memoryTypeCount; ++i)
        if ((type_bits & (1u << i)) &&
            (c->mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    /* UINT32_MAX (not 0) so callers fail loudly rather than silently allocating from the wrong heap;
     * every caller guards the result before passing it to vkAllocateMemory. */
    fprintf(stderr, "no suitable memory type\n");
    return UINT32_MAX;
}

int vkc_create_buffer(VkCtx *c, VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props, VkBuffer *out_buf, VkDeviceMemory *out_mem) {
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VKCHECK(vkCreateBuffer(c->device, &bi, NULL, out_buf));

    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c->device, *out_buf, &mr);
    VkMemoryAllocateFlagsInfo fi = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ai.pNext = &fi;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = vkc_find_memory_type(c, mr.memoryTypeBits, props);
    if (ai.memoryTypeIndex == UINT32_MAX) return 0;
    VKCHECK(vkAllocateMemory(c->device, &ai, NULL, out_mem));
    VKCHECK(vkBindBufferMemory(c->device, *out_buf, *out_mem, 0));
    return 1;
}

int vkc_create_host_buffer(VkCtx *c, VkDeviceSize size, VkBufferUsageFlags usage,
                           VkBuffer *out_buf, VkDeviceMemory *out_mem, void **out_ptr) {
    if (!vkc_create_buffer(c, size, usage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            out_buf, out_mem)) return 0;
    VKCHECK(vkMapMemory(c->device, *out_mem, 0, size, 0, out_ptr));
    return 1;
}

VkDeviceAddress vkc_buffer_address(VkCtx *c, VkBuffer buf) {
    VkBufferDeviceAddressInfo ai = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    ai.buffer = buf;
    return vkGetBufferDeviceAddress(c->device, &ai);
}

void vkc_destroy_buffer(VkCtx *c, VkBuffer buf, VkDeviceMemory mem) {
    if (buf) vkDestroyBuffer(c->device, buf, NULL);
    if (mem) vkFreeMemory(c->device, mem, NULL);
}

VkCommandBuffer vkc_single_time_begin(VkCtx *c) {
    VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = c->cmd_pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(c->device, &ai, &cmd);
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void vkc_single_time_end(VkCtx *c, VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkCommandBufferSubmitInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cbi.commandBuffer = cmd;
    VkSubmitInfo2 si = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    si.commandBufferInfoCount = 1; si.pCommandBufferInfos = &cbi;
    vkQueueSubmit2(c->queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(c->queue);
    vkFreeCommandBuffers(c->device, c->cmd_pool, 1, &cmd);
}

int vkc_create_texture_rgba16f(VkCtx *c, uint32_t w, uint32_t h, const void *data,
                               VkPipelineStageFlags2 consumer_stages,
                               VkImage *out_img, VkDeviceMemory *out_mem, VkImageView *out_view) {
    VkDeviceSize bytes = (VkDeviceSize)w * h * 8;  /* RGBA16F = 8 bytes/texel */

    VkImageCreateInfo ii = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ii.extent.width = w; ii.extent.height = h; ii.extent.depth = 1;
    ii.mipLevels = 1; ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VKCHECK(vkCreateImage(c->device, &ii, NULL, out_img));

    VkMemoryRequirements mr; vkGetImageMemoryRequirements(c->device, *out_img, &mr);
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = vkc_find_memory_type(c, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX) return 0;
    VKCHECK(vkAllocateMemory(c->device, &ai, NULL, out_mem));
    VKCHECK(vkBindImageMemory(c->device, *out_img, *out_mem, 0));

    /* staging upload */
    VkBuffer staging; VkDeviceMemory staging_mem; void *p;
    if (!vkc_create_host_buffer(c, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &staging, &staging_mem, &p)) return 0;
    memcpy(p, data, (size_t)bytes);

    VkCommandBuffer cmd = vkc_single_time_begin(c);
    image_barrier(cmd, *out_img, VK_IMAGE_ASPECT_COLOR_BIT,
                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region = { 0 };
    region.bufferRowLength = w;                 /* tightly packed: row width in texels */
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width = w; region.imageExtent.height = h; region.imageExtent.depth = 1;
    vkCmdCopyBufferToImage(cmd, staging, *out_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    image_barrier(cmd, *out_img, VK_IMAGE_ASPECT_COLOR_BIT,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  consumer_stages, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    vkc_single_time_end(c, cmd);
    vkc_destroy_buffer(c, staging, staging_mem);

    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = *out_img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1; vi.subresourceRange.layerCount = 1;
    VKCHECK(vkCreateImageView(c->device, &vi, NULL, out_view));
    return 1;
}

/* ------------------------------------------------------------------ shader objects */

int vkc_create_graphics_shaders(VkCtx *c,
        const uint32_t *vs, size_t vs_bytes, const uint32_t *fs, size_t fs_bytes,
        const VkDescriptorSetLayout *set_layouts, uint32_t set_count,
        const VkPushConstantRange *push_range,
        VkShaderEXT *out_vert, VkShaderEXT *out_frag) {
    VkShaderCreateInfoEXT info[2];
    memset(info, 0, sizeof info);
    for (int i = 0; i < 2; ++i) {
        info[i].sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
        info[i].flags = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT;
        info[i].codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
        info[i].pName = "main";
        info[i].setLayoutCount = set_count;
        info[i].pSetLayouts = set_layouts;
        info[i].pushConstantRangeCount = push_range ? 1 : 0;
        info[i].pPushConstantRanges = push_range;
    }
    info[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    info[0].nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
    info[0].pCode = vs; info[0].codeSize = vs_bytes;
    info[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    info[1].pCode = fs; info[1].codeSize = fs_bytes;

    VkShaderEXT shaders[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VKCHECK(vkCreateShadersEXT(c->device, 2, info, NULL, shaders));
    *out_vert = shaders[0];
    *out_frag = shaders[1];
    return 1;
}

int vkc_create_vertex_shader(VkCtx *c, const uint32_t *vs, size_t vs_bytes,
        const VkDescriptorSetLayout *set_layouts, uint32_t set_count,
        const VkPushConstantRange *push_range, VkShaderEXT *out_vert) {
    /* UNLINKED (no LINK_STAGE bit) so it is bindable on its own with the fragment stage left NULL -
     * used for rasterizer-discard vertex-only timing, where no fragment shader runs. */
    VkShaderCreateInfoEXT info = { VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT };
    info.stage = VK_SHADER_STAGE_VERTEX_BIT;
    info.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
    info.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
    info.pName = "main";
    info.pCode = vs; info.codeSize = vs_bytes;
    info.setLayoutCount = set_count;
    info.pSetLayouts = set_layouts;
    info.pushConstantRangeCount = push_range ? 1 : 0;
    info.pPushConstantRanges = push_range;
    VkShaderEXT sh = VK_NULL_HANDLE;
    VKCHECK(vkCreateShadersEXT(c->device, 1, &info, NULL, &sh));
    *out_vert = sh;
    return 1;
}

void vkc_destroy_shader(VkCtx *c, VkShaderEXT s) {
    if (s) vkDestroyShaderEXT(c->device, s, NULL);
}

void vkc_bind_shaders(VkCtx *c, VkCommandBuffer cmd, VkShaderEXT vert, VkShaderEXT frag) {
    (void)c;
    VkShaderStageFlagBits stages[2] = { VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT };
    VkShaderEXT shaders[2] = { vert, frag };
    vkCmdBindShadersEXT(cmd, 2, stages, shaders);
}

void vkc_set_default_state(VkCtx *c, VkCommandBuffer cmd, VkExtent2D extent,
        const VkVertexInputBindingDescription2EXT *bindings, uint32_t nb,
        const VkVertexInputAttributeDescription2EXT *attrs, uint32_t na) {
    (void)c;
    VkViewport vp = { 0, 0, (float)extent.width, (float)extent.height, 0.0f, 1.0f };
    VkRect2D sc = { { 0, 0 }, extent };
    vkCmdSetViewportWithCount(cmd, 1, &vp);
    vkCmdSetScissorWithCount(cmd, 1, &sc);
    vkCmdSetLineWidth(cmd, 1.0f);
    vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);
    vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
    vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    vkCmdSetDepthTestEnable(cmd, VK_TRUE);
    vkCmdSetDepthWriteEnable(cmd, VK_TRUE);
    vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_LESS);
    vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
    vkCmdSetDepthBiasEnable(cmd, VK_FALSE);
    vkCmdSetStencilTestEnable(cmd, VK_FALSE);
    vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    vkCmdSetPrimitiveRestartEnable(cmd, VK_FALSE);
    vkCmdSetVertexInputEXT(cmd, nb, bindings, na, attrs);
    vkCmdSetPolygonModeEXT(cmd, VK_POLYGON_MODE_FILL);
    vkCmdSetRasterizationSamplesEXT(cmd, VK_SAMPLE_COUNT_1_BIT);
    VkSampleMask mask = 0xFFFFFFFFu;
    vkCmdSetSampleMaskEXT(cmd, VK_SAMPLE_COUNT_1_BIT, &mask);
    vkCmdSetAlphaToCoverageEnableEXT(cmd, VK_FALSE);
    vkCmdSetLogicOpEnableEXT(cmd, VK_FALSE);
    VkBool32 blend = VK_FALSE;
    vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blend);
    VkColorComponentFlags mask_rgba = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    vkCmdSetColorWriteMaskEXT(cmd, 0, 1, &mask_rgba);
}

/* ------------------------------------------------------------------ image barrier helper */

static void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                          VkImageLayout old_layout, VkImageLayout new_layout,
                          VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                          VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkImageMemoryBarrier2 b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    b.srcStageMask = src_stage; b.srcAccessMask = src_access;
    b.dstStageMask = dst_stage; b.dstAccessMask = dst_access;
    b.oldLayout = old_layout; b.newLayout = new_layout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange.aspectMask = aspect;
    b.subresourceRange.levelCount = 1; b.subresourceRange.layerCount = 1;
    VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
}

/* ------------------------------------------------------------------ GPU timestamps */

/* Read this slot's prior-frame timestamps (NO wait - the submission already completed before this
 * is called). Only zones in the slot's written mask are read; deltas are taken modulo the queue's
 * valid-bit width so a wrapped counter never produces garbage. Results land in gpu_zone_ms +
 * gpu_results_valid for vkc_gpu_results_ms to publish. */
static void read_gpu_results(VkCtx *c) {
    c->gpu_results_valid = 0;
    if (!c->timestamp_valid_bits) return;
    uint32_t mask = c->gpu_zone_written[c->cur_frame];
    if (!mask) return;
    /* consume the mask now: if begin_frame early-returns (swapchain rebuild) before recording a new
     * frame, the retry must not re-read and re-publish these same completed timestamps. */
    c->gpu_zone_written[c->cur_frame] = 0;

    uint64_t data[VKC_GPU_QUERIES_PER_FRAME * 2];   /* value + availability per query */
    VkResult r = vkGetQueryPoolResults(c->device, c->gpu_qpool[c->cur_frame],
            0, VKC_GPU_QUERIES_PER_FRAME, sizeof data, data,
            2 * sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (r != VK_SUCCESS && r != VK_NOT_READY) return;

    /* valid==64 must not do `1<<64` (UB); use a full mask there. */
    uint64_t vmask = (c->timestamp_valid_bits >= 64) ? ~0ull
                                                     : ((1ull << c->timestamp_valid_bits) - 1ull);
    uint32_t valid = 0;
    for (uint32_t z = 0; z < VKC_GPU_ZONE_COUNT; ++z) {
        if (!(mask & (1u << z))) continue;
        uint32_t qb = 2u * z, qe = 2u * z + 1u;
        uint64_t bv = data[qb * 2], ba = data[qb * 2 + 1];
        uint64_t ev = data[qe * 2], ea = data[qe * 2 + 1];
        if (!ba || !ea) continue;                   /* end not recorded/ready this frame */
        uint64_t delta = (ev - bv) & vmask;
        c->gpu_zone_ms[z] = (float)((double)delta * (double)c->timestamp_period_ns * 1e-6);
        valid |= (1u << z);
    }
    c->gpu_results_valid = valid;
}

void vkc_gpu_zone_begin(VkCtx *c, VkCommandBuffer cmd, vkc_gpu_zone zone) {
    if (!c->timestamp_valid_bits) return;
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                         c->gpu_qpool[c->cur_frame], 2u * zone);
    c->gpu_zone_written[c->cur_frame] |= 1u << zone;
}

void vkc_gpu_zone_end(VkCtx *c, VkCommandBuffer cmd, vkc_gpu_zone zone) {
    if (!c->timestamp_valid_bits) return;
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                         c->gpu_qpool[c->cur_frame], 2u * zone + 1u);
}

uint32_t vkc_gpu_results_ms(VkCtx *c, float out_ms[VKC_GPU_ZONE_COUNT]) {
    for (uint32_t z = 0; z < VKC_GPU_ZONE_COUNT; ++z) out_ms[z] = c->gpu_zone_ms[z];
    return c->gpu_results_valid;
}

/* ------------------------------------------------------------------ per-frame */

int vkc_begin_frame(VkCtx *c, VkCommandBuffer *out_cmd, VkExtent2D *out_extent) {
    c->cur_frame = (uint32_t)(c->frame_index % VKC_FRAMES_IN_FLIGHT);
    c->wait_ms = 0.0; c->acquire_ms = 0.0;

    /* throttle: wait until this slot's previous submission finished (GPU backpressure - timed, but
     * the demo keeps it OUT of the CPU work metric). */
    if (c->frame_index >= VKC_FRAMES_IN_FLIGHT) {
        VkSemaphoreWaitInfo wi = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
        uint64_t wait_val = c->frame_timeline[c->cur_frame];
        wi.semaphoreCount = 1; wi.pSemaphores = &c->timeline; wi.pValues = &wait_val;
        double t0 = prof_now_s();
        vkWaitSemaphores(c->device, &wi, UINT64_MAX);
        c->wait_ms = (prof_now_s() - t0) * 1000.0;

        /* The slot's prior submission is now complete - consume its timestamps BEFORE any early
         * return (swapchain rebuild / acquire failure), so the next reset starts from a clean pool. */
        read_gpu_results(c);
    }

    if (c->framebuffer_resized) { c->framebuffer_resized = 0; recreate_swapchain(c); return 0; }

    uint32_t image_index = 0;
    double ta = prof_now_s();
    VkResult r = vkAcquireNextImageKHR(c->device, c->swapchain, UINT64_MAX,
                                       c->image_available[c->cur_frame], VK_NULL_HANDLE, &image_index);
    c->acquire_ms = (prof_now_s() - ta) * 1000.0;
    if (r == VK_ERROR_OUT_OF_DATE_KHR) { recreate_swapchain(c); return 0; }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) { fprintf(stderr, "acquire failed %d\n", (int)r); return 0; }
    c->cur_image = image_index;

    VkCommandBuffer cmd = c->cmd[c->cur_frame];
    c->cur_cmd = cmd;
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    /* Reset this slot's query pool (illegal inside a render pass, so do it here) and stamp the
     * frame-begin timestamp before the barriers / BeginRendering. */
    if (c->timestamp_valid_bits) {
        vkCmdResetQueryPool(cmd, c->gpu_qpool[c->cur_frame], 0, VKC_GPU_QUERIES_PER_FRAME);
        c->gpu_zone_written[c->cur_frame] = 0;
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             c->gpu_qpool[c->cur_frame], 2u * VKC_GPU_ZONE_FRAME);
        c->gpu_zone_written[c->cur_frame] |= 1u << VKC_GPU_ZONE_FRAME;
    }

    image_barrier(cmd, c->images[image_index], VK_IMAGE_ASPECT_COLOR_BIT,
                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    image_barrier(cmd, c->depth_image, VK_IMAGE_ASPECT_DEPTH_BIT,
                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo color = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    color.imageView = c->image_views[image_index];
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color.float32[0] = c->clear_rgb[0];
    color.clearValue.color.float32[1] = c->clear_rgb[1];
    color.clearValue.color.float32[2] = c->clear_rgb[2];
    color.clearValue.color.float32[3] = 1.0f;

    VkRenderingAttachmentInfo depth = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depth.imageView = c->depth_view;
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.clearValue.depthStencil.depth = 1.0f;

    VkRenderingInfo ri = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea.extent = c->extent;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    ri.pDepthAttachment = &depth;
    vkCmdBeginRendering(cmd, &ri);

    *out_cmd = cmd;
    *out_extent = c->extent;
    return 1;
}

void vkc_request_screenshot(VkCtx *c, const char *path) { c->pending_screenshot = path; }

void vkc_end_frame(VkCtx *c) {
    VkCommandBuffer cmd = c->cur_cmd;
    vkCmdEndRendering(cmd);

    int do_capture = c->pending_screenshot && c->can_capture;
    VkBuffer rb = VK_NULL_HANDLE; VkDeviceMemory rbm = VK_NULL_HANDLE; void *rbptr = NULL;
    if (do_capture) {
        VkDeviceSize sz = (VkDeviceSize)c->extent.width * c->extent.height * 4;
        if (!vkc_create_host_buffer(c, sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb, &rbm, &rbptr)) {
            fprintf(stderr, "[demo] screenshot readback alloc failed; skipping capture\n");
            do_capture = 0; c->pending_screenshot = NULL;
        }
    }
    if (do_capture) {
        image_barrier(cmd, c->images[c->cur_image], VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy region = { 0 };
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent.width = c->extent.width;
        region.imageExtent.height = c->extent.height;
        region.imageExtent.depth = 1;
        vkCmdCopyImageToBuffer(cmd, c->images[c->cur_image], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &region);
        image_barrier(cmd, c->images[c->cur_image], VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                      VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                      VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
    } else {
        image_barrier(cmd, c->images[c->cur_image], VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
    }

    /* frame-end timestamp (bottom of pipe - after all rendering work) */
    if (c->timestamp_valid_bits) {
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                             c->gpu_qpool[c->cur_frame], 2u * VKC_GPU_ZONE_FRAME + 1u);
    }
    vkEndCommandBuffer(cmd);

    uint64_t signal_value = ++c->timeline_value;
    VkSemaphoreSubmitInfo wait = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    wait.semaphore = c->image_available[c->cur_frame];
    wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSemaphoreSubmitInfo signals[2];
    memset(signals, 0, sizeof signals);
    signals[0].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signals[0].semaphore = c->render_finished[c->cur_image];
    signals[0].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    signals[1].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signals[1].semaphore = c->timeline;
    signals[1].value = signal_value;
    signals[1].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cbi.commandBuffer = cmd;
    VkSubmitInfo2 si = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    si.waitSemaphoreInfoCount = 1; si.pWaitSemaphoreInfos = &wait;
    si.commandBufferInfoCount = 1; si.pCommandBufferInfos = &cbi;
    si.signalSemaphoreInfoCount = 2; si.pSignalSemaphoreInfos = signals;
    vkQueueSubmit2(c->queue, 1, &si, VK_NULL_HANDLE);
    c->frame_timeline[c->cur_frame] = signal_value;

    VkPresentInfoKHR pi = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &c->render_finished[c->cur_image];
    pi.swapchainCount = 1;
    pi.pSwapchains = &c->swapchain;
    pi.pImageIndices = &c->cur_image;
    VkResult r = vkQueuePresentKHR(c->queue, &pi);

    if (do_capture) {
        vkQueueWaitIdle(c->queue);
        uint32_t w = c->extent.width, h = c->extent.height;
        int bgra = (c->color_format == VK_FORMAT_B8G8R8A8_SRGB || c->color_format == VK_FORMAT_B8G8R8A8_UNORM);
        uint8_t *px = (uint8_t *)malloc((size_t)w * h * 4);
        const uint8_t *src = (const uint8_t *)rbptr;
        for (size_t i = 0; i < (size_t)w * h; ++i) {
            px[i*4+0] = bgra ? src[i*4+2] : src[i*4+0];
            px[i*4+1] = src[i*4+1];
            px[i*4+2] = bgra ? src[i*4+0] : src[i*4+2];
            px[i*4+3] = 255;
        }
        png_write_rgba(c->pending_screenshot, w, h, px);
        fprintf(stderr, "[demo] screenshot -> %s (%ux%u)\n", c->pending_screenshot, w, h);
        free(px);
        vkc_destroy_buffer(c, rb, rbm);
        c->pending_screenshot = NULL;
    }

    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || c->framebuffer_resized) {
        c->framebuffer_resized = 0;
        recreate_swapchain(c);
    }
    c->frame_index++;
}
