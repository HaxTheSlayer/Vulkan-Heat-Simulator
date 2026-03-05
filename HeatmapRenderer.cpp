#include "HeatmapRenderer.h"
#include <fstream>
#include <array>

// --- Constructor ---
HeatmapRenderer::HeatmapRenderer(VulkanContext& context, const PDESolver& solver)
    : vkContext(context), pdeSolver(solver)
{
    createDescriptorSets();
    createGraphicsPipeline();
}

void HeatmapRenderer::createDescriptorSets() {
    vk::DescriptorSetLayoutBinding layoutBinding(
        0,
        vk::DescriptorType::eStorageBuffer,
        1,
        vk::ShaderStageFlagBits::eFragment,
        nullptr
    );

    vk::DescriptorSetLayoutCreateInfo layoutInfo(
        {},                                                 //flags
        1,                                                  //bindingCount
        &layoutBinding                                      //pBindings
    );
    descriptorSetLayout = vk::raii::DescriptorSetLayout(vkContext.getDevice(), layoutInfo);

    std::array poolSize{
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 2)
    };
    vk::DescriptorPoolCreateInfo poolInfo(
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,       //flags
        2,                                                          //maxSets
        poolSize.size(),                                            //poolSizeCount
        poolSize.data()                                             //pPoolSizes
    );
    descriptorPool = vk::raii::DescriptorPool(vkContext.getDevice(), poolInfo);

    std::vector<vk::DescriptorSetLayout> layouts(2, *descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo(
        *descriptorPool,        // descriptorPool
        2,                      // descriptorSetCount
        layouts.data()          // pSetLayouts (Pointer to the array!)
    );

    descriptorSets.clear();
    descriptorSets = vkContext.getDevice().allocateDescriptorSets(allocInfo);

    for (int i = 0; i < 2; i++) {
        vk::DescriptorBufferInfo bufferInfo(
            *pdeSolver.getStorageBuffers()[i],  // buffer
            0,                                  // offset
            VK_WHOLE_SIZE                       // range
        );

        vk::WriteDescriptorSet descriptorWrite(
            *descriptorSets[i],                 // dstSet
            0,                                  // dstBinding
            0,                                  // dstArrayElement
            1,                                  // descriptorCount
            vk::DescriptorType::eStorageBuffer, // descriptorType
            nullptr,                            // pImageInfo
            &bufferInfo,                        // pBufferInfo
            nullptr                             // pTexelBufferView
        );

        vkContext.getDevice().updateDescriptorSets(descriptorWrite, {});
    }
}

void HeatmapRenderer::createGraphicsPipeline() {

    std::vector <char> vert = readFile("shaders/vert.spv");
    std::vector <char> frag = readFile("shaders/frag.spv");

    vk::ShaderModuleCreateInfo vertInfo({}, vert.size(), reinterpret_cast<const uint32_t*>(vert.data()));
    vk::raii::ShaderModule vertModule(vkContext.getDevice(), vertInfo);

    vk::ShaderModuleCreateInfo fragInfo({}, frag.size(), reinterpret_cast<const uint32_t*>(frag.data()));
    vk::raii::ShaderModule fragModule(vkContext.getDevice(), fragInfo);

    vk::PipelineShaderStageCreateInfo shaderStages[] = {
        { {}, vk::ShaderStageFlagBits::eVertex, *vertModule, "main" },
        { {}, vk::ShaderStageFlagBits::eFragment, *fragModule, "main" }
    };

    //Using Full Screen Triangle trick, so vertex input not needed
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
        {},                                         //flags
        vk::PrimitiveTopology::eTriangleList,       //topology
        vk::False                                   //primitiveRestartEnable
    );


    vk::PipelineViewportStateCreateInfo viewportState( 
        {},         //flags
        1,          //viewportCount
        nullptr,
        1,          //scissorCount
        nullptr
    );

    vk::PipelineRasterizationStateCreateInfo rasterizer(
        {},                             // flags
        vk::False,                      // depthClampEnable
        vk::False,                      // rasterizerDiscardEnable
        vk::PolygonMode::eFill,         // polygonMode
        vk::CullModeFlagBits::eNone,    // cullMode
        vk::FrontFace::eClockwise,      // frontFace
        vk::False,                      // depthBiasEnable
        0.0f,                           // depthBiasConstantFactor 
        0.0f,                           // depthBiasClamp 
        0.0f,                           // depthBiasSlopeFactor 
        1.0f                            // lineWidth
    );

    vk::PipelineMultisampleStateCreateInfo multisampling(
        {},                                 //flags
        vk::SampleCountFlagBits::e1,        //rasterizationSamples
        vk::False                           //sampleShadingEnable
    );


    vk::PipelineColorBlendAttachmentState colorBlendAttachment(
        vk::False,                                  //blendEnable
        vk::BlendFactor::eSrcAlpha,                 //srcColorBlendFactor
        vk::BlendFactor::eOneMinusSrcAlpha,         //dstColorBlendFactor
        vk::BlendOp::eAdd,                          //colorBlendOp
        vk::BlendFactor::eOneMinusSrcAlpha,         //srcAlphaBlendFactor
        vk::BlendFactor::eZero,                     //dstAlphaBlendFactor
        vk::BlendOp::eAdd,                          //alphaBlendOp
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA   //colorWriteMask
    );

    vk::PipelineColorBlendStateCreateInfo colorBlending(
        {},                         //flags
        vk::False,                  //logicOpEnable
        vk::LogicOp::eCopy,         //logicOp
        1,                          //attachmentCount
        &colorBlendAttachment       //pAttachments
    );

    vk::DynamicState dynamicStates[] = {
		    vk::DynamicState::eViewport,
		    vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState(
        {},                     //flags
        2,                      //dynamicStateCount
        dynamicStates           //pDynamicStates
    );

    vk::PushConstantRange pushConstantRange(
        vk::ShaderStageFlagBits::eFragment, // stageFlags
        0,                                  // offset
        sizeof(uint32_t) * 2                // size (8 bytes total)
    );

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 
        *descriptorSetLayout, 
        pushConstantRange
    );
    pipelineLayout = vk::raii::PipelineLayout(vkContext.getDevice(), pipelineLayoutInfo);

    vk::Format colorFormat = vkContext.getSwapChainSurfaceFormat().format;
    vk::PipelineRenderingCreateInfo renderingCreateInfo(
        0, 
        1, 
        &colorFormat, 
        vk::Format::eUndefined, 
        vk::Format::eUndefined
    );

    vk::GraphicsPipelineCreateInfo pipelineInfo(
        {}, 
        2, 
        shaderStages, 
        &vertexInputInfo, 
        &inputAssembly, 
        nullptr, 
        &viewportState,
        &rasterizer, 
        &multisampling, 
        nullptr, 
        &colorBlending, 
        &dynamicState, 
        *pipelineLayout
    );

    pipelineInfo.pNext = &renderingCreateInfo;

    graphicsPipeline = vk::raii::Pipeline(vkContext.getDevice(), nullptr, pipelineInfo);
}

void HeatmapRenderer::recordDrawCommands(vk::raii::CommandBuffer& cmdBuffer, uint32_t imageIndex, uint32_t currentFrame) {
    const auto& swapChainImageViews = vkContext.getSwapChainImageViews();
    vk::Extent2D swapChainExtent = vkContext.getSwapChainExtent();

    cmdBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    cmdBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

    vk::ClearValue clearColor;
    clearColor.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});

    vk::RenderingAttachmentInfo attachmentInfo(
        *swapChainImageViews[imageIndex],               // imageView
        vk::ImageLayout::eColorAttachmentOptimal,       // imageLayout
        vk::ResolveModeFlagBits::eNone,                 // resolveMode 
        nullptr,                                        // resolveImageView
        vk::ImageLayout::eUndefined,                    // resolveImageLayout 
        vk::AttachmentLoadOp::eClear,                   // loadOp
        vk::AttachmentStoreOp::eStore,                  // storeOp
        clearColor                                      // clearValue
    );
    vk::RenderingInfo renderingInfo(
        {},                                             // flags 
        vk::Rect2D({ 0, 0 }, swapChainExtent),          // renderArea
        1,                                              // layerCount
        0,                                              // viewMask 
        1,                                              // colorAttachmentCount
        &attachmentInfo                                 // pColorAttachments
    );

    cmdBuffer.beginRendering(renderingInfo);
    cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
    cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, { *descriptorSets[currentFrame] }, {});
    uint32_t dimensions[2] = { swapChainExtent.width, swapChainExtent.height };
    cmdBuffer.pushConstants<uint32_t>(*pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, dimensions);
    cmdBuffer.draw(3, 1, 0, 0);
    cmdBuffer.endRendering();
}

std::vector<char> HeatmapRenderer::readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file!");
    }
    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();

    return buffer;
}