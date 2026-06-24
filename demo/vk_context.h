/* Vulkan 1.4 context for the demo: instance/device/swapchain/depth/sync, plus the per-frame
 * dynamic-rendering loop and a few buffer/shader-object helpers. Modern feature set only -
 * dynamic rendering, synchronization2, timeline semaphores, buffer device address, descriptor
 * indexing, and VK_EXT_shader_object (probed; pipeline fallback is a documented TODO if absent).
 *
 * This is demo plumbing; the marrow runtime contains no GPU code. */
#ifndef DEMO_VK_CONTEXT_H
#define DEMO_VK_CONTEXT_H

#include <volk.h>
#include <stddef.h>
#include <stdint.h>

typedef struct GLFWwindow GLFWwindow;

#define VKC_FRAMES_IN_FLIGHT 2
#define VKC_MAX_IMAGES       8

/* GPU timing zones (timestamp queries). FRAME brackets the whole frame; the others bracket the
 * matching draws. Order MUST match prof_gpu_zone (profiler.h). Keep VKC_GPU_ZONE_COUNT last. */
typedef enum {
    VKC_GPU_ZONE_FRAME = 0,
    VKC_GPU_ZONE_GROUND,
    VKC_GPU_ZONE_CROWD,
    VKC_GPU_ZONE_HEROES,
    VKC_GPU_ZONE_SKEL,      /* bone-line render LOD (far tail + global all-skeleton near) */
    VKC_GPU_ZONE_HUD,
    VKC_GPU_ZONE_COUNT
} vkc_gpu_zone;
#define VKC_GPU_QUERIES_PER_FRAME (2u * VKC_GPU_ZONE_COUNT)   /* begin + end per zone */

typedef struct {
    GLFWwindow *win;
    VkInstance  instance;
    VkDebugUtilsMessengerEXT debug;
    VkSurfaceKHR surface;
    VkPhysicalDevice phys;
    VkPhysicalDeviceMemoryProperties mem_props;
    uint32_t    queue_family;
    VkDevice    device;
    VkQueue     queue;
    int         shader_object_ok;     /* VK_EXT_shader_object usable */

    /* swapchain */
    VkSwapchainKHR swapchain;
    VkFormat       color_format;
    VkColorSpaceKHR color_space;
    VkPresentModeKHR present_mode;
    VkExtent2D     extent;
    uint32_t       image_count;
    VkImage        images[VKC_MAX_IMAGES];
    VkImageView    image_views[VKC_MAX_IMAGES];
    VkSemaphore    render_finished[VKC_MAX_IMAGES];  /* per-image binary; signaled by submit */

    /* depth */
    VkFormat       depth_format;
    VkImage        depth_image;
    VkDeviceMemory depth_mem;
    VkImageView    depth_view;

    /* per-frame */
    VkCommandPool  cmd_pool;
    VkCommandBuffer cmd[VKC_FRAMES_IN_FLIGHT];
    VkSemaphore    image_available[VKC_FRAMES_IN_FLIGHT]; /* binary; signaled by acquire */
    VkSemaphore    timeline;                              /* monotonic; CPU throttle */
    uint64_t       timeline_value;
    uint64_t       frame_timeline[VKC_FRAMES_IN_FLIGHT];  /* value signaled per slot */
    uint64_t       frame_index;

    int            framebuffer_resized;
    int            can_capture;            /* swapchain supports TRANSFER_SRC */
    const char    *pending_screenshot;     /* captured + written at next end_frame */

    /* GPU timestamps: per-frame-in-flight query pool (begin+end per zone). Disabled (n/a) when
     * the graphics queue reports timestampValidBits == 0. */
    VkQueryPool    gpu_qpool[VKC_FRAMES_IN_FLIGHT];
    float          timestamp_period_ns;    /* limits.timestampPeriod (ns/tick)              */
    uint32_t       timestamp_valid_bits;   /* graphics queue timestampValidBits; 0 => disabled */
    uint32_t       gpu_zone_written[VKC_FRAMES_IN_FLIGHT]; /* zones recorded this slot (begin bit) */
    float          gpu_zone_ms[VKC_GPU_ZONE_COUNT];        /* latest readback (ms) per zone   */
    uint32_t       gpu_results_valid;      /* mask of zones with a fresh ms this frame       */

    /* CPU spans measured inside begin_frame (ms) - the demo's profiler reads these so the timeline
     * wait and image acquire show up as separate scopes from the work. */
    double         wait_ms, acquire_ms;

    /* state during a begin/end frame */
    uint32_t       cur_frame;   /* frame_index % FRAMES_IN_FLIGHT */
    uint32_t       cur_image;   /* acquired swapchain image index */
    VkCommandBuffer cur_cmd;

    float          clear_rgb[3];
} VkCtx;

int  vkc_init(VkCtx *c, GLFWwindow *win);
void vkc_destroy(VkCtx *c);
void vkc_wait_idle(VkCtx *c);

/* Begin a frame: throttle on the timeline, acquire, begin the command buffer, barrier the
 * swapchain color + depth into attachment layouts, vkCmdBeginRendering, set full-extent
 * viewport/scissor. Returns 1 with *out_cmd set, or 0 if the frame was skipped (swapchain
 * rebuilt / minimized) - caller should `continue`. */
int  vkc_begin_frame(VkCtx *c, VkCommandBuffer *out_cmd, VkExtent2D *out_extent);
/* End: vkCmdEndRendering, barrier color->present, submit2 (signal binary + timeline), present. */
void vkc_end_frame(VkCtx *c);

/* Capture the frame being submitted by the NEXT vkc_end_frame to a PNG at `path` (swapchain
 * readback). `path` must stay alive until then. Used for headless visual verification. */
void vkc_request_screenshot(VkCtx *c, const char *path);

/* GPU timestamp zones. Bracket a draw with begin/end (TOP_OF_PIPE for begin, BOTTOM_OF_PIPE for
 * end) inside the BeginRendering scope; the FRAME zone is written automatically by begin/end_frame.
 * No-ops when timestamps are unsupported. */
void vkc_gpu_zone_begin(VkCtx *c, VkCommandBuffer cmd, vkc_gpu_zone zone);
void vkc_gpu_zone_end  (VkCtx *c, VkCommandBuffer cmd, vkc_gpu_zone zone);
/* Copy the latest per-zone GPU times (ms) into out_ms[VKC_GPU_ZONE_COUNT]; returns the bitmask of
 * zones that have a fresh result this frame (0 when unsupported or not yet available). */
uint32_t vkc_gpu_results_ms(VkCtx *c, float out_ms[VKC_GPU_ZONE_COUNT]);

/* Memory / buffers. */
uint32_t vkc_find_memory_type(VkCtx *c, uint32_t type_bits, VkMemoryPropertyFlags props);
int  vkc_create_buffer(VkCtx *c, VkDeviceSize size, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags props, VkBuffer *out_buf, VkDeviceMemory *out_mem);
void vkc_destroy_buffer(VkCtx *c, VkBuffer buf, VkDeviceMemory mem);
/* Host-visible+coherent buffer, persistently mapped (returns the mapping in *out_ptr). */
int  vkc_create_host_buffer(VkCtx *c, VkDeviceSize size, VkBufferUsageFlags usage,
                            VkBuffer *out_buf, VkDeviceMemory *out_mem, void **out_ptr);
VkDeviceAddress vkc_buffer_address(VkCtx *c, VkBuffer buf);

/* Single-time command submission (synchronous; waits idle). For uploads/transitions. */
VkCommandBuffer vkc_single_time_begin(VkCtx *c);
void            vkc_single_time_end(VkCtx *c, VkCommandBuffer cmd);

/* Create a sampled RGBA16F 2D texture and upload `data` (w*h*8 bytes) via a staging buffer,
 * leaving it in SHADER_READ_ONLY_OPTIMAL. Used for the baked GPU palette.
 * `consumer_stages` is the pipeline-stage scope the texture is sampled from (e.g.
 * VERTEX_SHADER for the crowd draw, COMPUTE_SHADER for the validation dispatch) - the
 * post-upload barrier makes the transfer write visible to exactly those stages. */
int vkc_create_texture_rgba16f(VkCtx *c, uint32_t w, uint32_t h, const void *data,
                               VkPipelineStageFlags2 consumer_stages,
                               VkImage *out_img, VkDeviceMemory *out_mem, VkImageView *out_view);

/* Shader objects: create a LINKED vertex+fragment pair sharing one set-layout/push-constant
 * "layout". Returns 1 on success. */
int  vkc_create_graphics_shaders(VkCtx *c,
        const uint32_t *vs, size_t vs_bytes, const uint32_t *fs, size_t fs_bytes,
        const VkDescriptorSetLayout *set_layouts, uint32_t set_count,
        const VkPushConstantRange *push_range,   /* may be NULL */
        VkShaderEXT *out_vert, VkShaderEXT *out_frag);
/* Standalone (unlinked) vertex shader object - bindable with the fragment stage left NULL, for
 * rasterizer-discard vertex-only timing. */
int  vkc_create_vertex_shader(VkCtx *c, const uint32_t *vs, size_t vs_bytes,
        const VkDescriptorSetLayout *set_layouts, uint32_t set_count,
        const VkPushConstantRange *push_range, VkShaderEXT *out_vert);
void vkc_destroy_shader(VkCtx *c, VkShaderEXT s);
void vkc_bind_shaders(VkCtx *c, VkCommandBuffer cmd, VkShaderEXT vert, VkShaderEXT frag);

/* Set the full dynamic-state vector a shader-object draw requires (depth test on, back-face
 * cull, triangle list, the given vertex input). Call once per draw before vkCmdDraw*. */
void vkc_set_default_state(VkCtx *c, VkCommandBuffer cmd, VkExtent2D extent,
        const VkVertexInputBindingDescription2EXT *bindings, uint32_t nb,
        const VkVertexInputAttributeDescription2EXT *attrs, uint32_t na);

#endif /* DEMO_VK_CONTEXT_H */
