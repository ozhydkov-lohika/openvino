// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "tile/tile_kernel_selector.h"
#include "tile/tile_kernel_ref.h"
#include "intel_gpu/runtime/error_handler.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct tile_impl : typed_primitive_impl_ocl<tile> {
    using parent = typed_primitive_impl_ocl<tile>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<tile_impl>(*this);
    }

public:
    static primitive_impl* create(const tile_node& arg) {
        auto tile_params = get_default_params<kernel_selector::tile_params>(arg);
        auto tile_optional_params =
            get_default_optional_params<kernel_selector::tile_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::tile_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(tile_params, tile_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto tile = new tile_impl(arg, best_kernels[0]);

        return tile;
    }
};

namespace detail {

attach_tile_impl::attach_tile_impl() {
    const auto types = {data_types::i8, data_types::u8, data_types::i32, data_types::f16, data_types::f32};
    const auto formats = {
            format::bfyx,
            format::bfzyx,
            format::bfwzyx,
            format::b_fs_zyx_fsv16,
            format::b_fs_zyx_fsv32,
            format::b_fs_yx_fsv16,
            format::b_fs_yx_fsv32,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_yx_bsv32_fsv32,
            format::bs_fs_zyx_bsv16_fsv32,
            format::bs_fs_zyx_bsv16_fsv16,
            format::bs_fs_zyx_bsv32_fsv32,
            format::bs_fs_zyx_bsv32_fsv16
    };

    std::set<std::tuple<data_types, format::type>> keys;
    for (const auto& t : types) {
        for (const auto& f : formats) {
            keys.emplace(t, f);
        }
    }
    implementation_map<tile>::add(impl_types::ocl, tile_impl::create, keys);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
