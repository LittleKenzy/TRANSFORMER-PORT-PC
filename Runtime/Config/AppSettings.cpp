#include "AppSettings.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace tf::ps2 {

namespace fs = std::filesystem;

std::string AppSettings::default_config_path() {
#ifdef _WIN32
    if (const char* local = std::getenv("LOCALAPPDATA")) {
        return (fs::path(local) / "TransformersNative" / "settings.ini").string();
    }
#endif
    return (fs::temp_directory_path() / "TransformersNative" / "settings.ini").string();
}

bool AppSettings::load(const std::string& path) {
    std::string p = path.empty() ? default_config_path() : path;
    std::ifstream ifs(p);
    if (!ifs) return false;

    std::string line;
    while (std::getline(ifs, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq);
        std::string v = line.substr(eq + 1);
        try {
            if (k == "last_iso_path") last_iso_path = v;
            else if (k == "last_module_path") last_module_path = v;
            else if (k == "memory_card_dir") memory_card_dir = v;
            else if (k == "cache_dir") cache_dir = v;
            else if (k == "internal_resolution_scale") graphics.internal_resolution_scale = std::stoul(v);
            else if (k == "vsync") graphics.vsync = (v == "1" || v == "true");
            else if (k == "bilinear_filtering") graphics.bilinear_filtering = (v == "1" || v == "true");
            else if (k == "aspect_ratio") graphics.aspect_ratio = static_cast<AspectRatioMode>(std::stoi(v));
            else if (k == "deadzone") input.deadzone = std::stof(v);
            else if (k == "sensitivity") input.sensitivity = std::stof(v);
            else if (k == "invert_camera_x") input.invert_camera_x = (v == "1" || v == "true");
            else if (k == "invert_camera_y") input.invert_camera_y = (v == "1" || v == "true");
            else if (k == "vibration_enabled") input.vibration_enabled = (v == "1" || v == "true");
            else if (k == "audio_enabled") audio_enabled = (v == "1" || v == "true");
            else if (k == "audio_volume_percent") audio_volume_percent = std::stoul(v);
            else if (k == "start_fullscreen") start_fullscreen = (v == "1" || v == "true");
            else if (k == "show_touch_controls") show_touch_controls = (v == "1" || v == "true");
        } catch (...) {
        }
    }
    return true;
}

bool AppSettings::save(const std::string& path) const {
    std::string p = path.empty() ? default_config_path() : path;
    std::error_code ec;
    fs::create_directories(fs::path(p).parent_path(), ec);
    std::ofstream ofs(p, std::ios::trunc);
    if (!ofs) return false;

    ofs << "last_iso_path=" << last_iso_path << "\n";
    ofs << "last_module_path=" << last_module_path << "\n";
    ofs << "memory_card_dir=" << memory_card_dir << "\n";
    ofs << "cache_dir=" << cache_dir << "\n";
    ofs << "internal_resolution_scale=" << graphics.internal_resolution_scale << "\n";
    ofs << "vsync=" << (graphics.vsync ? 1 : 0) << "\n";
    ofs << "bilinear_filtering=" << (graphics.bilinear_filtering ? 1 : 0) << "\n";
    ofs << "aspect_ratio=" << static_cast<int>(graphics.aspect_ratio) << "\n";
    ofs << "deadzone=" << input.deadzone << "\n";
    ofs << "sensitivity=" << input.sensitivity << "\n";
    ofs << "invert_camera_x=" << (input.invert_camera_x ? 1 : 0) << "\n";
    ofs << "invert_camera_y=" << (input.invert_camera_y ? 1 : 0) << "\n";
    ofs << "vibration_enabled=" << (input.vibration_enabled ? 1 : 0) << "\n";
    ofs << "audio_enabled=" << (audio_enabled ? 1 : 0) << "\n";
    ofs << "audio_volume_percent=" << audio_volume_percent << "\n";
    ofs << "start_fullscreen=" << (start_fullscreen ? 1 : 0) << "\n";
    ofs << "show_touch_controls=" << (show_touch_controls ? 1 : 0) << "\n";
    return ofs.good();
}

} // namespace tf::ps2
