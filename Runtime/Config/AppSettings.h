#pragma once

#include "../../Recompiler/Common/Types.h"
#include "../../Graphics/Common/GraphicsBackend.h"
#include "../Input/DualShock2.h"
#include <string>

namespace tf::ps2 {

struct AppSettings {
    std::string last_iso_path;
    std::string last_module_path;
    std::string memory_card_dir;
    std::string cache_dir;

    GraphicsSettings graphics;
    InputConfig input;

    bool audio_enabled = true;
    u32 audio_volume_percent = 100;
    bool start_fullscreen = false;
    bool show_touch_controls = true;

    static std::string default_config_path();
    bool load(const std::string& path = "");
    bool save(const std::string& path = "") const;
};

} // namespace tf::ps2
