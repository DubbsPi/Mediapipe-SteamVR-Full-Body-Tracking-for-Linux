#include "openvr_driver.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <cmath> // Required for sqrt, acos, etc.
#include <cstring>

// Linux-specific socket includes
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>

using namespace vr;

// =================================================================
// VR Controller Device
// =================================================================
class VRControllerDevice : public ITrackedDeviceServerDriver {
private:
    uint32_t device_index_ = k_unTrackedDeviceIndexInvalid;
    std::string serial_;
    DriverPose_t pose_;
    bool is_left_hand_;

    // Input handles
    VRInputComponentHandle_t trigger_click_component_ = k_ulInvalidInputComponentHandle;
    VRInputComponentHandle_t trigger_value_component_ = k_ulInvalidInputComponentHandle;
    VRInputComponentHandle_t grip_component_ = k_ulInvalidInputComponentHandle;
    VRInputComponentHandle_t system_component_ = k_ulInvalidInputComponentHandle;
    VRInputComponentHandle_t menu_component_ = k_ulInvalidInputComponentHandle;
    VRInputComponentHandle_t trackpad_x_component_ = k_ulInvalidInputComponentHandle;
    VRInputComponentHandle_t trackpad_y_component_ = k_ulInvalidInputComponentHandle;
    VRInputComponentHandle_t trackpad_click_component_ = k_ulInvalidInputComponentHandle;

public:
    VRControllerDevice(const std::string& serial, bool is_left)
    : serial_(serial), is_left_hand_(is_left) {
        pose_ = {};
        pose_.poseIsValid = true;
        pose_.result = TrackingResult_Running_OK;
        pose_.deviceIsConnected = true;

        // FIX: HmdQuaternion_t has 4 members (w, x, y, z), not 5.
        pose_.qWorldFromDriverRotation = { 1.0, 0.0, 0.0, 0.0 };
        pose_.qDriverFromHeadRotation = { 1.0, 0.0, 0.0, 0.0 };

        pose_.vecPosition[0] = is_left ? -0.2 : 0.2; // Initial position
        pose_.vecPosition[1] = 1.0;
        pose_.vecPosition[2] = -0.3;
    }

    EVRInitError Activate(uint32_t unObjectId) override {
        device_index_ = unObjectId;

        // --- Property registration ---
        VRProperties()->SetStringProperty(device_index_, Prop_SerialNumber_String, serial_.c_str());
        VRProperties()->SetBoolProperty(device_index_, Prop_WillDriftInYaw_Bool, false);
        VRProperties()->SetStringProperty(device_index_, Prop_RenderModelName_String, "vr_controller_vive_1_5");
        VRProperties()->SetInt32Property(device_index_, Prop_DeviceClass_Int32, TrackedDeviceClass_Controller);
        VRProperties()->SetInt32Property(device_index_, Prop_ControllerRoleHint_Int32, is_left_hand_ ? TrackedControllerRole_LeftHand : TrackedControllerRole_RightHand);

        // --- Input component creation ---
        VRDriverInput()->CreateBooleanComponent(device_index_, "/input/trigger/click", &trigger_click_component_);
        VRDriverInput()->CreateScalarComponent(device_index_, "/input/trigger/value", &trigger_value_component_, VRScalarType_Absolute, VRScalarUnits_NormalizedOneSided);
        VRDriverInput()->CreateBooleanComponent(device_index_, "/input/grip/click", &grip_component_);
        VRDriverInput()->CreateBooleanComponent(device_index_, "/input/system/click", &system_component_);
        VRDriverInput()->CreateBooleanComponent(device_index_, "/input/application_menu/click", &menu_component_);
        VRDriverInput()->CreateBooleanComponent(device_index_, "/input/trackpad/click", &trackpad_click_component_);
        VRDriverInput()->CreateScalarComponent(device_index_, "/input/trackpad/x", &trackpad_x_component_, VRScalarType_Absolute, VRScalarUnits_NormalizedTwoSided);
        VRDriverInput()->CreateScalarComponent(device_index_, "/input/trackpad/y", &trackpad_y_component_, VRScalarType_Absolute, VRScalarUnits_NormalizedTwoSided);

        return VRInitError_None;
    }

    void Deactivate() override {
        device_index_ = k_unTrackedDeviceIndexInvalid;
    }

    void EnterStandby() override {}

    void* GetComponent(const char* pchComponentNameAndVersion) override {
        return nullptr;
    }

    void DebugRequest(const char* pchRequest, char* pchResponseBuffer, uint32_t unResponseBufferSize) override {}

    DriverPose_t GetPose() override {
        return pose_;
    }

    // Update pose with new position and rotation data
    void UpdatePose(double x, double y, double z, double qw, double qx, double qy, double qz) {
        pose_.vecPosition[0] = x;
        pose_.vecPosition[1] = y;
        pose_.vecPosition[2] = z;
        pose_.qRotation = { qw, qx, qy, qz };

        if (device_index_ != k_unTrackedDeviceIndexInvalid) {
            VRServerDriverHost()->TrackedDevicePoseUpdated(device_index_, pose_, sizeof(DriverPose_t));
        }
    }
};

// =================================================================
// Generic Tracker Device
// =================================================================
class TrackerDevice : public ITrackedDeviceServerDriver {
private:
    uint32_t device_index_ = k_unTrackedDeviceIndexInvalid;
    std::string serial_;
    DriverPose_t pose_;
    std::string role_;

public:
    TrackerDevice(const std::string& serial, const std::string& role)
    : serial_(serial), role_(role) {
        pose_ = {};
        pose_.poseIsValid = true;
        pose_.result = TrackingResult_Running_OK;
        pose_.deviceIsConnected = true;
        // FIX: HmdQuaternion_t has 4 members (w, x, y, z), not 5.
        pose_.qWorldFromDriverRotation = { 1.0, 0.0, 0.0, 0.0 };
        pose_.qDriverFromHeadRotation = { 1.0, 0.0, 0.0, 0.0 };
    }

    EVRInitError Activate(uint32_t unObjectId) override {
        device_index_ = unObjectId;

        VRProperties()->SetStringProperty(device_index_, Prop_SerialNumber_String, serial_.c_str());
        VRProperties()->SetStringProperty(device_index_, Prop_ModelNumber_String, "MediaPipe Tracker");
        VRProperties()->SetStringProperty(device_index_, Prop_RenderModelName_String, "{htc}vr_tracker_vive_1_0"); // A default tracker model

        std::cout << "Tracker activated: " << serial_ << " as " << role_ << " (ID: " << device_index_ << ")" << std::endl;
        return VRInitError_None;
    }

    void Deactivate() override {
        device_index_ = k_unTrackedDeviceIndexInvalid;
    }

    void EnterStandby() override {}

    void* GetComponent(const char* pchComponentNameAndVersion) override {
        return nullptr;
    }

    void DebugRequest(const char* pchRequest, char* pchResponseBuffer, uint32_t unResponseBufferSize) override {}

    DriverPose_t GetPose() override {
        return pose_;
    }

    void UpdatePose(double x, double y, double z, double qw, double qx, double qy, double qz) {
        pose_.vecPosition[0] = x;
        pose_.vecPosition[1] = y;
        pose_.vecPosition[2] = z;
        pose_.qRotation = { qw, qx, qy, qz };

        if (device_index_ != k_unTrackedDeviceIndexInvalid) {
            VRServerDriverHost()->TrackedDevicePoseUpdated(device_index_, pose_, sizeof(DriverPose_t));
        }
    }
};

// =================================================================
// Main Driver Provider
// =================================================================
class MediaPipeDriver : public IServerTrackedDeviceProvider {
private:
    std::unique_ptr<VRControllerDevice> left_controller_;
    std::unique_ptr<VRControllerDevice> right_controller_;
    std::map<int, std::unique_ptr<TrackerDevice>> trackers_;
    std::thread communication_thread_;
    std::atomic<bool> running_{ false };
    int server_fd_ = -1; // Keep track of the server socket

    // --- FIX: Added helper structs for 3D math ---
    struct Vec3 {
        float x = 0.0f, y = 0.0f, z = 0.0f;

        float length() const { return std::sqrt(x * x + y * y + z * z); }

        Vec3 normalize() const {
            float len = length();
            if (len > 1e-6f) return {x / len, y / len, z / len};
            return {0.0f, 0.0f, 0.0f};
        }

        Vec3 operator-(const Vec3& other) const { return {x - other.x, y - other.y, z - other.z}; }
        float dot(const Vec3& other) const { return x * other.x + y * other.y + z * other.z; }
        Vec3 cross(const Vec3& other) const {
            return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
        }
    };

    struct Quaternion {
        float w = 1.0f, x = 0.0f, y = 0.0f, z = 0.0f;
    };

public:
    EVRInitError Init(IVRDriverContext* pDriverContext) override {
        VR_INIT_SERVER_DRIVER_CONTEXT(pDriverContext);
        std::cout << "MediaPipe Driver initializing..." << std::endl;

        left_controller_ = std::make_unique<VRControllerDevice>("MP_LeftController", true);
        right_controller_ = std::make_unique<VRControllerDevice>("MP_RightController", false);
        VRServerDriverHost()->TrackedDeviceAdded("MP_LeftController", TrackedDeviceClass_Controller, left_controller_.get());
        VRServerDriverHost()->TrackedDeviceAdded("MP_RightController", TrackedDeviceClass_Controller, right_controller_.get());

        // Create trackers with their roles
        CreateTracker(11, "MP_LeftShoulder");
        CreateTracker(12, "MP_RightShoulder");
        CreateTracker(13, "MP_LeftElbow");
        CreateTracker(14, "MP_RightElbow");
        CreateTracker(15, "MP_LeftWrist");
        CreateTracker(16, "MP_RightWrist");
        CreateTracker(17, "MP_LeftPinky");
        CreateTracker(18, "MP_RightPinky");
        CreateTracker(19, "MP_LeftIndex");
        CreateTracker(20, "MP_RightIndex");
        CreateTracker(21, "MP_LeftThumb");
        CreateTracker(22, "MP_RightThumb");
        CreateTracker(23, "MP_LeftHip");
        CreateTracker(24, "MP_RightHip");
        CreateTracker(25, "MP_LeftKnee");
        CreateTracker(26, "MP_RightKnee");
        CreateTracker(27, "MP_LeftAnkle");
        CreateTracker(28, "MP_RightAnkle");
        CreateTracker(29, "MP_LeftHeel");
        CreateTracker(30, "MP_RightHeel");
        CreateTracker(31, "MP_LeftFootIndex");
        CreateTracker(32, "MP_RightFootIndex");

        running_ = true;
        communication_thread_ = std::thread(&MediaPipeDriver::CommunicationLoop, this);

        return VRInitError_None;
    }

    void Cleanup() override {
        running_ = false;

        // Shutdown for the communication thread
        // Closing the server socket will cause `accept()` to fail, unblocking the thread.
        if (server_fd_ != -1) {
            shutdown(server_fd_, SHUT_RDWR);
            close(server_fd_);
            server_fd_ = -1;
        }

        if (communication_thread_.joinable()) {
            communication_thread_.join();
        }

        left_controller_.reset();
        right_controller_.reset();
        trackers_.clear();
        VR_CLEANUP_SERVER_DRIVER_CONTEXT();
    }

    const char* const* GetInterfaceVersions() override { return k_InterfaceVersions; }
    void RunFrame() override {}
    bool ShouldBlockStandbyMode() override { return false; }
    void EnterStandby() override {}
    void LeaveStandby() override {}

private:
    // Rotation calculation logic
    static Quaternion getLineRotationQuaternion(const Vec3& point1, const Vec3& point2, const Vec3& referenceDir = {0, 0, 1}) {
        Vec3 lineDirection = (point2 - point1).normalize();
        Vec3 refDir = referenceDir.normalize();

        float dot = refDir.dot(lineDirection);

        if (std::abs(dot - 1.0f) < 1e-6f) { // Already aligned
            return {1, 0, 0, 0};
        }
        if (std::abs(dot + 1.0f) < 1e-6f) { // 180-degree rotation
            Vec3 perpendicular = std::abs(refDir.x) < 0.9f ? Vec3{1, 0, 0}.cross(refDir) : Vec3{0, 1, 0}.cross(refDir);
            perpendicular = perpendicular.normalize();
            return {0, perpendicular.x, perpendicular.y, perpendicular.z};
        }

        Vec3 axis = refDir.cross(lineDirection).normalize();
        float angle = std::acos(dot) * 1.5f;
        float halfAngle = angle * 0.75;
        float sinHalf = std::sin(halfAngle);

        return {std::cos(halfAngle), axis.x * sinHalf, axis.y * sinHalf, axis.z * sinHalf};
    }

    void CreateTracker(int id, const std::string& role) {
        std::string serial = "MP_Tracker_" + std::to_string(id);
        auto tracker = std::make_unique<TrackerDevice>(serial, role);
        VRServerDriverHost()->TrackedDeviceAdded(serial.c_str(), TrackedDeviceClass_GenericTracker, tracker.get());
        trackers_[id] = std::move(tracker);
    }

    void CommunicationLoop() {
        // Store elbow positions as Vec3
        Vec3 l_finger_pos, r_finger_pos;

        const char* socket_path = "/tmp/vr_unix_socket.sock";
        unlink(socket_path); // Remove old socket file if it exists

        server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
        if (server_fd_ == -1) {
            std::cerr << "Failed to create UNIX socket" << std::endl;
            return;
        }

        struct sockaddr_un addr = {};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

        if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
            std::cerr << "Failed to bind UNIX socket" << std::endl;
            close(server_fd_);
            return;
        }

        if (listen(server_fd_, 1) == -1) {
            std::cerr << "Failed to listen on UNIX socket" << std::endl;
            close(server_fd_);
            return;
        }

        std::cout << "Waiting for Python connection on " << socket_path << std::endl;

        while (running_) {
            int client_fd = accept(server_fd_, nullptr, nullptr);
            if (client_fd == -1) {
                if (running_) std::cerr << "Failed to accept connection" << std::endl;
                continue;
            }

            std::cout << "Python connected!" << std::endl;

            // Expected packet: 1 int32_t (ID) + 3 floats (x, y, z)
            const size_t PACKET_SIZE = sizeof(int32_t) + 3 * sizeof(float);
            char buffer[PACKET_SIZE];

            while (running_) {
                ssize_t bytes_received = recv(client_fd, buffer, PACKET_SIZE, MSG_WAITALL);

                if (bytes_received == PACKET_SIZE) {
                    // This data unpacking logic is correct for handling network byte order
                    int32_t network_id;
                    memcpy(&network_id, buffer, sizeof(int32_t));
                    int32_t id = ntohl(network_id);

                    float pos[3];
                    for (int i = 0; i < 3; ++i) {
                        uint32_t network_float_as_int;
                        memcpy(&network_float_as_int, buffer + sizeof(int32_t) + (i * sizeof(float)), sizeof(uint32_t));
                        uint32_t host_float_as_int = ntohl(network_float_as_int);
                        memcpy(&pos[i], &host_float_as_int, sizeof(float));
                    }

                    auto it = trackers_.find(id);
                    if (it != trackers_.end()) {
                        // Update tracker with identity rotation
                        it->second->UpdatePose(pos[0], pos[1], pos[2], 1, 0, 0, 0);
                    }

                    // Store elbow positions for controller rotation
                    if (id == 19) { // Left index finger
                        l_finger_pos = {pos[0], pos[1], pos[2]};
                    } else if (id == 20) { // Right index finger
                        r_finger_pos = {pos[0], pos[1], pos[2]};
                    }

                    // Update controllers if it's a wrist landmark
                    if (id == 15) { // Left Wrist
                        Vec3 wrist_pos = {pos[0], pos[1], pos[2]};
                        // Call quaternion logic and update controller pose
                        Quaternion l_rotation = getLineRotationQuaternion(l_finger_pos, wrist_pos);
                        left_controller_->UpdatePose(pos[0] * 2.0f, pos[1], pos[2] * 2.0f, l_rotation.w, l_rotation.x, l_rotation.y, l_rotation.z);

                    } else if (id == 16) { // Right Wrist
                        Vec3 wrist_pos = {pos[0], pos[1], pos[2]};
                        Quaternion r_rotation = getLineRotationQuaternion(r_finger_pos, wrist_pos);
                        right_controller_->UpdatePose(pos[0] * 2.0f, pos[1], pos[2] * 2.0f, r_rotation.w, r_rotation.x, r_rotation.y, r_rotation.z);
                    }

                } else {
                    std::cout << "Connection lost or error." << std::endl;
                    break; // Exit inner loop and wait for a new connection
                }
            }

            close(client_fd);
            if(running_) std::cout << "Python disconnected. Waiting for new connection..." << std::endl;
        }

        // Final cleanup of the server socket
        if (server_fd_ != -1) close(server_fd_);
        unlink(socket_path);
    }
};

// =================================================================
// Driver Factory
// =================================================================
MediaPipeDriver g_driver;

extern "C" __attribute__((visibility("default"))) void* HmdDriverFactory(const char* interface_name, int* return_code) {
    if (strcmp(interface_name, IServerTrackedDeviceProvider_Version) == 0) {
        return &g_driver;
    }
    if (return_code) *return_code = VRInitError_Init_InterfaceNotFound;
    return nullptr;
}
