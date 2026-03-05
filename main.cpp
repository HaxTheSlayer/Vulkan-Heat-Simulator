#include "VulkanContext.h"
#include "PDESolver.h"
#include "HeatmapRenderer.h"
#include <iostream>
#include <chrono>

int main() {
    try {
        VulkanContext context(800, 800);
        PDESolver solver(context, 800, 800);
        HeatmapRenderer renderer(context, solver);

        vk::raii::Device& device = context.getDevice();
        vk::raii::Queue& queue = context.getQueue();

        vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);
        vk::raii::Fence inFlightFence(device, fenceInfo);

        vk::SemaphoreCreateInfo semaphoreInfo{};
        vk::raii::Semaphore imageAvailableSemaphore(device, semaphoreInfo);
        vk::raii::Semaphore renderFinishedSemaphore(device, semaphoreInfo);

        vk::CommandBufferAllocateInfo allocInfo(*context.getCommandPool(), vk::CommandBufferLevel::ePrimary, 1);
        vk::raii::CommandBuffer cmdBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());

        auto lastTime = std::chrono::high_resolution_clock::now();

        while (!glfwWindowShouldClose(context.getWindow())) {
            glfwPollEvents();

            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
            lastTime = currentTime;

            // STEP 1: Wait & Reset
            if (device.waitForFences({ *inFlightFence }, vk::True, UINT64_MAX) != vk::Result::eSuccess) {
                throw std::runtime_error("failed to wait for fence!");
            }
            device.resetFences({ *inFlightFence });

            // STEP 2: Acquire Image (FIX: Pass the semaphore, not the fence!)
            auto [result, imageIndex] = context.getSwapChain().acquireNextImage(UINT64_MAX, *imageAvailableSemaphore, nullptr);

            // --- BEGIN COMMAND RECORDING ---
            cmdBuffer.reset();
            cmdBuffer.begin({});

            // STEP 3: Dispatch Compute
            double xpos, ypos;
            glfwGetCursorPos(context.getWindow(), &xpos, &ypos);
            int state = glfwGetMouseButton(context.getWindow(), GLFW_MOUSE_BUTTON_LEFT);

            solver.dispatchCompute(cmdBuffer, deltaTime, static_cast<float>(xpos), static_cast<float>(ypos), state == GLFW_PRESS);

            // STEP 4: Compute-to-Graphics Barrier (FIX: Use standard vk::MemoryBarrier)
            vk::MemoryBarrier memoryBarrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead
            );
            cmdBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eFragmentShader,
                {}, { memoryBarrier }, {}, {}
            );

            // STEP 5: Layout Transition (Undefined -> ColorAttachment)
            vk::ImageMemoryBarrier barrier2(
                {}, vk::AccessFlagBits::eColorAttachmentWrite,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                context.getSwapChainImages()[imageIndex],
                { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
            );
            cmdBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                {}, {}, {}, { barrier2 }
            );

            // STEP 6: Execute Renderer
            renderer.recordDrawCommands(cmdBuffer, imageIndex, solver.getCurrentFrame());

            // STEP 7: Layout Transition (ColorAttachment -> PresentSrcKHR)
            vk::ImageMemoryBarrier barrier3(
                vk::AccessFlagBits::eColorAttachmentWrite, {},
                vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                context.getSwapChainImages()[imageIndex],
                { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
            );
            cmdBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe,
                {}, {}, {}, { barrier3 }
            );
            cmdBuffer.end();

            // STEP 8: Submit to Queue (FIX: Raw pointers using &*)
            vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
            vk::SubmitInfo computeSubmitInfo(
                1, &*imageAvailableSemaphore, waitStages,
                1, &*cmdBuffer,
                1, &*renderFinishedSemaphore
            );
            queue.submit({ computeSubmitInfo }, *inFlightFence); // FIX: Pass the fence!

            vk::PresentInfoKHR presentInfo(
                1, &*renderFinishedSemaphore,
                1, &*context.getSwapChain(),
                &imageIndex
            );
            queue.presentKHR(presentInfo);
            queue.waitIdle();
        }

        // STEP 10: Clean up
        device.waitIdle();

    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}