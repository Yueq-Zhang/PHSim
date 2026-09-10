#pragma once

#include <filesystem>
#include <string>

namespace phsim {

struct ResolvedConfigPath {
    std::filesystem::path path;
    bool used_legacy_working_directory = false;
};

// References stored in a configuration file are owned by that file. Resolve
// relative references from the owner's directory so a simulation is
// independent of the directory from which the executable was launched.
//
// Older PHSim configurations used process-working-directory-relative paths.
// Keep a compatibility fallback only when the owner-relative target does not
// exist and the legacy target does.
inline ResolvedConfigPath ResolveConfigPath(
    const std::string& referenced_path, const std::string& owner_config_path) {
    if (referenced_path.empty()) {
        return {};
    }

    const std::filesystem::path reference(referenced_path);
    if (reference.is_absolute()) {
        return {reference.lexically_normal(), false};
    }

    const std::filesystem::path owner =
        std::filesystem::absolute(owner_config_path).lexically_normal();
    const std::filesystem::path owner_relative =
        (owner.parent_path() / reference).lexically_normal();

    std::error_code owner_error;
    if (std::filesystem::exists(owner_relative, owner_error) && !owner_error) {
        return {owner_relative, false};
    }

    const std::filesystem::path legacy =
        std::filesystem::absolute(reference).lexically_normal();
    std::error_code legacy_error;
    if (std::filesystem::exists(legacy, legacy_error) && !legacy_error) {
        return {legacy, true};
    }

    // Return the canonical owner-relative interpretation even when the target
    // is absent. The eventual loader then reports the standardized path.
    return {owner_relative, false};
}

}  // namespace phsim
