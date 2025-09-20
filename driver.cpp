#include "openvr_driver.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <chrono>

using namespace vr;

// Simple tracker device class
class TrackerDevice : public ITrackedDeviceServerDriver {
private:
    uint32_t device_index_ = k_unTrackedDeviceIndexInvalid;
    std::string serial_;
    DriverPose_t pose_;
    std::atomic<bool> pose_updated_{false};

public:
    TrackerDevice(const std::string& serial) : serial_(serial) {
        // Initialize pose
        pose_ = {};
        pose_.poseIsValid = true;
        pose_.result = TrackingResult_Running_OK;
        pose_.deviceIsConnected = true;

        // Set initial position (origin)
        pose_.qWorldFromDriverRotation.w = 1.0;
        pose_.qDriverFromHeadRotation.w = 1.0;
        pose_.vecPosition[0] = 0.0;
        pose_.vecPosition[1] = 1.0;  // 1 meter up
        pose_.vecPosition[2] = 0.0;
    }

    // ITrackedDeviceServerDriver methods
    EVRInitError Activate(uint32_t unObjectId) override {
        device_index_ = unObjectId;

        // Set up device properties
        auto props = VRProperties();
        props->SetStringProperty(device_index_, Prop_ModelNumber_String, "MediaPipe Tracker");
        props->SetStringProperty(device_index_, Prop_SerialNumber_String, serial_.c_str());
        props->SetInt32Property(device_index_, Prop_DeviceClass_Int32, TrackedDeviceClass_GenericTracker);

        std::cout << "Tracker activated: " << serial_ << " (ID: " << device_index_ << ")" << std::endl;
        return VRInitError_None;
    }

    void Deactivate() override {
        device_index_ = k_unTrackedDeviceIndexInvalid;
    }

    void EnterStandby() override {}
    void* GetComponent(const char* pchComponentNameAndVersion) override { return nullptr; }
    void DebugRequest(const char* pchRequest, char* pchResponseBuffer, uint32_t unResponseBufferSize) override {}

    DriverPose_t GetPose() override {
        return pose_;
    }

    // Custom method to update pose from Python
    void UpdatePose(double x, double y, double z, double qw, double qx, double qy, double qz) {
        pose_.vecPosition[0] = x;
        pose_.vecPosition[1] = y;
        pose_.vecPosition[2] = z;

        pose_.qRotation.w = qw;
        pose_.qRotation.x = qx;
        pose_.qRotation.y = qy;
        pose_.qRotation.z = qz;

        pose_.poseTimeOffset = 0.0;
        pose_updated_ = true;

        // Tell SteamVR the pose updated
        if (device_index_ != k_unTrackedDeviceIndexInvalid) {
            VRServerDriverHost()->TrackedDevicePoseUpdated(device_index_, pose_, sizeof(DriverPose_t));
        }
    }
};

// Main driver class
class MediaPipeDriver : public IServerTrackedDeviceProvider {
private:
    std::vector<std::unique_ptr<TrackerDevice>> trackers_;
    std::thread communication_thread_;
    std::atomic<bool> running_{false};

    // Communication data structure
    struct TrackerData {
        int tracker_id;
        double x, y, z;
        double qw, qx, qy, qz;
    };

public:
    // IServerTrackedDeviceProvider methods
    EVRInitError Init(IVRDriverContext* pDriverContext) override {
        VR_INIT_SERVER_DRIVER_CONTEXT(pDriverContext);

        std::cout << "MediaPipe Driver initializing..." << std::endl;

        // Create all potential trackers at startup
        // MediaPipe body parts - adjust as needed
        trackers_.resize(34);  // Reserve space for 34 trackers

        // Basic trackers
        CreateTracker("MP_Waist", 0);
        CreateTracker("MP_LeftFoot", 1);
        CreateTracker("MP_RightFoot", 2);
        CreateTracker("MP_LeftHand", 3);
        CreateTracker("MP_RightHand", 4);
        CreateTracker("MP_LeftElbow", 5);
        CreateTracker("MP_RightElbow", 6);
        CreateTracker("MP_LeftKnee", 7);
        CreateTracker("MP_RightKnee", 8);
        CreateTracker("MP_Chest", 9);

        // Torso/spine detail
        CreateTracker("MP_LeftShoulder", 10);
        CreateTracker("MP_RightShoulder", 11);
        CreateTracker("MP_UpperChest", 12);
        CreateTracker("MP_Neck", 13);
        CreateTracker("MP_LeftHip", 14);
        CreateTracker("MP_RightHip", 15);

        // Left hand fingers
        CreateTracker("MP_LeftWrist", 16);
        CreateTracker("MP_LeftThumb", 17);
        CreateTracker("MP_LeftIndex", 18);
        CreateTracker("MP_LeftMiddle", 19);
        CreateTracker("MP_LeftRing", 20);
        CreateTracker("MP_LeftPinky", 21);

        // Right hand fingers
        CreateTracker("MP_RightWrist", 22);
        CreateTracker("MP_RightThumb", 23);
        CreateTracker("MP_RightIndex", 24);
        CreateTracker("MP_RightMiddle", 25);
        CreateTracker("MP_RightRing", 26);
        CreateTracker("MP_RightPinky", 27);

        // Additional leg detail
        CreateTracker("MP_LeftAnkle", 28);
        CreateTracker("MP_RightAnkle", 29);
        CreateTracker("MP_LeftHeel", 30);
        CreateTracker("MP_RightHeel", 31);
        CreateTracker("MP_LeftFootIndex", 32);
        CreateTracker("MP_RightFootIndex", 33);

        // Start communication thread
        running_ = true;
        communication_thread_ = std::thread(&MediaPipeDriver::CommunicationLoop, this);

        return VRInitError_None;
    }

    void Cleanup() override {
        running_ = false;
        if (communication_thread_.joinable()) {
            communication_thread_.join();
        }
        trackers_.clear();
    }

    const char* const* GetInterfaceVersions() override {
        return k_InterfaceVersions;
    }

    void RunFrame() override {
        // Called each frame by SteamVR
    }

    bool ShouldBlockStandbyMode() override { return false; }
    void EnterStandby() override {}
    void LeaveStandby() override {}

private:
    TrackerDevice* CreateTracker(const std::string& serial, int tracker_id) {
        auto tracker = std::make_unique<TrackerDevice>(serial);
        TrackerDevice* tracker_ptr = tracker.get();

        // Add to SteamVR
        VRServerDriverHost()->TrackedDeviceAdded(serial.c_str(), TrackedDeviceClass_GenericTracker, tracker.get());

        // Store in our vector
        trackers_[tracker_id] = std::move(tracker);

        std::cout << "Created tracker: " << serial << " (ID: " << tracker_id << ")" << std::endl;
        return tracker_ptr;
    }

    // Unix socket communication for Linux
    void CommunicationLoop() {
        const char* socket_path = "/tmp/mediapipe_vr.sock";

        // Clean up any existing socket file
        unlink(socket_path);

        int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (server_fd == -1) {
            std::cerr << "Failed to create socket" << std::endl;
            return;
        }

        struct sockaddr_un addr = {};
        addr.sun_family = AF_UNIX;
        strcpy(addr.sun_path, socket_path);

        if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
            std::cerr << "Failed to bind socket" << std::endl;
            close(server_fd);
            return;
        }

        if (listen(server_fd, 1) == -1) {
            std::cerr << "Failed to listen on socket" << std::endl;
            close(server_fd);
            return;
        }

        std::cout << "Waiting for Python connection on " << socket_path << std::endl;

        while (running_) {
            int client_fd = accept(server_fd, nullptr, nullptr);
            if (client_fd == -1) {
                if (running_) {
                    std::cerr << "Failed to accept connection" << std::endl;
                }
                continue;
            }

            std::cout << "Python connected!" << std::endl;

            TrackerData data;
            ssize_t bytes_read;

            while (running_) {
                bytes_read = read(client_fd, &data, sizeof(data));
                if (bytes_read == sizeof(data) && data.tracker_id < trackers_.size()) {
                    trackers_[data.tracker_id]->UpdatePose(
                        data.x, data.y, data.z,
                        data.qw, data.qx, data.qy, data.qz
                    );
                } else if (bytes_read <= 0) {
                    break; // Connection closed
                }
            }

            close(client_fd);
            std::cout << "Python disconnected" << std::endl;
        }

        close(server_fd);
        unlink(socket_path); // Clean up
    }
};

// Global driver instance
MediaPipeDriver g_driver;

// Required exports for SteamVR
extern "C" __attribute__((visibility("default"))) void* HmdDriverFactory(const char* interface_name, int* return_code) {
    if (strcmp(interface_name, IServerTrackedDeviceProvider_Version) == 0) {
        return &g_driver;
    }

    if (return_code) *return_code = VRInitError_Init_InterfaceNotFound;
    return nullptr;
}
