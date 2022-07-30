// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/tile.hpp>

#include <iostream>

using namespace cldnn;
using namespace ::tests;

template<typename data_t>
void tile_ref(const memory::ptr input, memory::ptr output, tile::tile_axis axis, int num_tiles)
{
    auto get_sizes = [](const layout& l, tile::tile_axis axis) -> std::pair<int, int>
    {
        switch (axis)
        {
            case tile::along_b: return std::make_pair(1, l.batch()*l.feature()*l.spatial(2)*l.spatial(1)*l.spatial(0));
            case tile::along_f: return std::make_pair(l.batch(), l.feature()*l.spatial(2)*l.spatial(1)*l.spatial(0));
            case tile::along_z: return std::make_pair(l.batch()*l.feature(), l.spatial(2)*l.spatial(1)*l.spatial(0));
            case tile::along_y: return std::make_pair(l.batch()*l.feature()*l.spatial(2), l.spatial(1)*l.spatial(0));
            case tile::along_x: return std::make_pair(l.batch()*l.feature()*l.spatial(2)*l.spatial(1), l.spatial(0));
            default: throw std::invalid_argument("Invalid axis(" + std::to_string(static_cast<int>(axis)) + ") in tile ref version");
        }
    };

    cldnn::mem_lock<data_t> src(input, get_test_stream());
    cldnn::mem_lock<data_t> dst(output, get_test_stream());

    const data_t* psrc = src.data();
    data_t* pdst = dst.data();

    auto sizes = get_sizes(input->get_layout(), axis);
    int outer_dim = sizes.first;
    int inner_dim = sizes.second;

    for (int i = 0; i < outer_dim; i++)
    {
        for (int t = 0; t < num_tiles; t++)
        {
            for (int j = 0; j < inner_dim; j++)
            {
                pdst[j] = psrc[j];
            }
            pdst += inner_dim;
        }
        psrc += inner_dim;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_b) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto output_ref = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(tile("tile", "input", tensor(2, 2, 2, 2)));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
                                     2.f, 0.f, 6.f, 5.2f };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, tile::along_b, 2);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_f) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto output_ref = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 4, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(tile("tile", "input", tensor(1, 4, 2, 2)));

    std::vector<float> input_vec = { 1.f, 0.f,
                                     5.f, 1.5f,

                                     2.f, 0.f,
                                     6.f, 5.2f };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, tile::along_f, 2);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_y) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto output_ref = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 4 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(tile("tile", "input", tensor(1, 2, 2, 4)));

    std::vector<float> input_vec = { 1.f, 0.f,
                                     5.f, 1.5f,

                                     2.f, 0.f,
                                     6.f, 5.2f };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, tile::along_y, 2);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_x) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto output_ref = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(tile("tile", "input", tensor(1, 2, 4, 2)));

    std::vector<float> input_vec = { 1.f, 0.f,
                                     5.f, 1.5f,

                                     2.f, 0.f,
                                     6.f, 5.2f };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, tile::along_x, 2);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_x_dense) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 2 } });
    auto output_ref = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(tile("tile", "input", tensor(1, 2, 4, 2)));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f};
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, tile::along_x, 4);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_z) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 1, 2, 2, 2, 2 } });
    auto output_ref = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 1, 2, 2, 2, 4 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(tile("tile", "input", tensor(1, 2, 2, 2, 4)));

    std::vector<float> input_vec = {
        1.f, 0.f,
        5.f, 1.5f,
        2.f, 0.f,
        6.f, 5.2f,
        1.f, 0.f,
        5.f, 1.5f,
        2.f, 0.f,
        6.f, 5.2f
    };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, tile::along_z, 2);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

namespace {
template<typename T>
struct Params {
    tensor input_tensor;
    std::vector<T> inputs;
    tensor output_tensor;
    std::vector<T> outputs;
    tile::tile_axis output_axis;
    int num_tiles;
};

template<typename T>
using ParamsWithLayout = std::tuple<
        Params<T>,
        format::type,   // source (plain) layout - bfyx or bfzyx
        format::type    // target (blocked) layout
>;

const std::vector<format::type> layouts_2d = {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32
};

const std::vector<format::type> layouts_3d = {
        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv16
};

template<typename T>
std::vector<T> getValues(const std::vector<float> &values) {
    std::vector<T> result(values.begin(), values.end());
    return result;
}

template<typename T>
std::vector<Params<T>> generateTileParams2D() {
    static const std::vector<Params<T>> result = {
            {
                    tensor(1, 2, 2, 2),
                    getValues<T>({1.f, 0.f, 5.f, 1.5f,
                                  2.f, 0.f, 6.f, 5.2f}),
                    tensor(1, 2, 2, 2),
                    getValues<T>({ 1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f, }),
                    tile::along_b,
                    2,
            },
            {
                    tensor(1, 2, 2, 2),
                    getValues<T>({1.f, 0.f,
                                  5.f, 1.5f,

                                  2.f, 0.f,
                                  6.f, 5.2f}),
                    tensor(1, 2, 2, 2),
                    getValues<T>({ 1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f, }),
                    tile::along_f,
                    2,
            },
            {
                    tensor(1, 2, 2, 2),
                    getValues<T>({1.f, 0.f,
                                  5.f, 1.5f,

                                  2.f, 0.f,
                                  6.f, 5.2f}),
                    tensor(1, 2, 2, 4),
                    getValues<T>({ 1.f, 0.f, 5.f, 1.5f, 1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f, 2.f, 0.f, 6.f, 5.2f, }),
                    tile::along_y,
                    2,
            },
            {
                    tensor(1, 2, 2, 2),
                    getValues<T>({1.f, 0.f,
                                  5.f, 1.5f,

                                  2.f, 0.f,
                                  6.f, 5.2f}),
                    tensor(1, 2, 4, 2),
                    getValues<T>({ 1.f, 0.f, 1.f, 0.f, 5.f, 1.5f, 5.f, 1.5f, 2.f, 0.f, 2.f, 0.f, 6.f, 5.2f, 6.f, 5.2f, }),
                    tile::along_x,
                    2,
            },
            {
                    tensor(1, 2, 1, 2),
                    getValues<T>({1.f, 0.f, 5.f, 1.5f}),
                    tensor(1, 2, 4, 2),
                    getValues<T>({ 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 5.f, 5.f, 5.f, 5.f, 1.5f, 1.5f, 1.5f, 1.5f, }),
                    tile::along_x,
                    4,
            },
    };
    return result;
}

template<typename T>
std::vector<Params<T>> generateTileParams3D() {
    static const std::vector<Params<T>> result = {
            {
                    {
                            tensor(1, 2, 2, 2, 2),
                            getValues<T>({
                                                 1.f, 0.f,
                                                 5.f, 1.5f,
                                                 2.f, 0.f,
                                                 6.f, 5.2f,
                                                 1.f, 0.f,
                                                 5.f, 1.5f,
                                                 2.f, 0.f,
                                                 6.f, 5.2f
                                         }),
                            tensor(1, 2, 2, 2, 2),
                            getValues<T>({ 1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f, 1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f, }),

                            tile::along_b,
                            2,
                    },
                    {
                            tensor(1, 2, 2, 2, 2),
                            getValues<T>({
                                                 1.f, 0.f,
                                                 5.f, 1.5f,
                                                 2.f, 0.f,
                                                 6.f, 5.2f,
                                                 1.f, 0.f,
                                                 5.f, 1.5f,
                                                 2.f, 0.f,
                                                 6.f, 5.2f
                                         }),
                            tensor(1, 2, 2, 2, 2),
                            getValues<T>({ 1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f, 1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f, }),

                            tile::along_f,
                            2,
                    },
                    {
                            tensor(1, 2, 2, 2, 2),
                            getValues<T>({
                                                 1.f, 0.f,
                                                 5.f, 1.5f,
                                                 2.f, 0.f,
                                                 6.f, 5.2f,
                                                 1.f, 0.f,
                                                 5.f, 1.5f,
                                                 2.f, 0.f,
                                                 6.f, 5.2f
                                         }),
                            tensor(1, 2, 4, 2, 2),
                            getValues<T>({ 1.f, 0.f, 1.f, 0.f, 5.f, 1.5f, 5.f, 1.5f, 2.f, 0.f, 2.f, 0.f, 6.f, 5.2f, 6.f, 5.2f, 1.f, 0.f, 1.f, 0.f, 5.f, 1.5f, 5.f, 1.5f, 2.f, 0.f, 2.f, 0.f, 6.f, 5.2f, 6.f, 5.2f, }),

                            tile::along_x,
                            2,
                    },
                    {
                            tensor(1, 2, 2, 2, 2),
                            getValues<T>({
                                                 1.f, 0.f,
                                                 5.f, 1.5f,
                                                 2.f, 0.f,
                                                 6.f, 5.2f,
                                                 1.f, 0.f,
                                                 5.f, 1.5f,
                                                 2.f, 0.f,
                                                 6.f, 5.2f
                                         }),
                            tensor(1, 2, 2, 4, 2),
                            getValues<T>({ 1.f, 0.f, 5.f, 1.5f, 1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f, 2.f, 0.f, 6.f, 5.2f, 1.f, 0.f, 5.f, 1.5f, 1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f, 2.f, 0.f, 6.f, 5.2f, }),

                            tile::along_y,
                            2,
                    },
                    {
                        tensor(1, 2, 2, 2, 2),
                                getValues<T>({
                                                     1.f, 0.f,
                                                     5.f, 1.5f,
                                                     2.f, 0.f,
                                                     6.f, 5.2f,
                                                     1.f, 0.f,
                                                     5.f, 1.5f,
                                                     2.f, 0.f,
                                                     6.f, 5.2f
                                             }),
                                tensor(1, 2, 2, 2, 4),
                                getValues<T>({ 1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f, 1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f, 1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f, 1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f, }),
                                tile::along_z,
                                2,
                    },
            }
    };
    return result;
}

struct PrintToStringParamName {
    template<class T>
    std::string operator()(const testing::TestParamInfo<ParamsWithLayout<T> > &param) {
        std::stringstream buf;
        Params<T> p;
        format::type plain_layout;
        format::type target_layout;
        std::tie(p, plain_layout, target_layout) = param.param;
        buf << " input tensor " << p.input_tensor.to_string()
            << " output tensor " << p.output_tensor.to_string()
            << " plain layout " << plain_layout
            << " target layout " << target_layout
            << " number Tiles " << p.num_tiles
            << " axis " << p.output_axis;
        return buf.str();
    }
};
};

template<typename T>
struct tile_test
        : public ::testing::TestWithParam<ParamsWithLayout<T> > {
public:
    void test() {
        const auto data_type = type_to_data_type<T>::value;
        Params<T> params;
        format::type plain_layout;
        format::type target_layout;

        std::tie(params, plain_layout, target_layout) = this->GetParam();

        const bool need_reorder = target_layout != plain_layout;

        auto& engine = get_test_engine();

        auto input = engine.allocate_memory({data_type, plain_layout, params.input_tensor});

        set_values(input, params.inputs);

        const std::string input_data_id = "input_id";
        topology topology;
        topology.add(input_layout(input_data_id, input->get_layout()));

        std::string input_id = input_data_id;
        if (need_reorder) {
            const std::string reorder_input_id = input_data_id + "_reordered";
            topology.add(reorder(reorder_input_id, input_data_id, target_layout, data_type));
            input_id = reorder_input_id;
        }

        const std::string result_data_id = "result_id";
        topology.add(tile(result_data_id, input_id, params.output_tensor));

        std::string result_id = result_data_id;
        if (need_reorder) {
            const primitive_id reorder_result_id = result_data_id + "_reordered";
            topology.add(reorder(reorder_result_id, result_data_id, plain_layout, data_type));
            result_id = reorder_result_id;
        }

        network network(engine, topology);

        network.set_input_data(input_data_id, input);

        auto result = network.execute();

        auto out_mem = result.at(result_id).get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(params.output_tensor.count(), out_ptr.size());

        auto output_ref = getOutputsRef(input);

        cldnn::mem_lock<T> output_ref_ptr(output_ref, get_test_stream());

        for (size_t i = 0; i < output_ref_ptr.size(); ++i) {
            EXPECT_NEAR(output_ref_ptr[i], out_ptr[i], 0.005) << "at i = " << i;
        }

        for (size_t i = 0; i < params.outputs.size(); ++i) {
            EXPECT_NEAR(params.outputs[i], out_ptr[i], 0.005) << "at i = " << i;
        }
    }

private:
    memory::ptr getOutputsRef(const memory::ptr input){
        Params<T> params;
        format::type plain_layout;
        format::type target_layout;
        std::tie(params, plain_layout, target_layout) = this->GetParam();
        format::type fmt = plain_layout;
        tensor output_size = params.output_tensor;
        tile::tile_axis output_axis = params.output_axis;
        int num_tiles = params.num_tiles;

        const auto data_type = type_to_data_type<T>::value;
        auto& engine = get_test_engine();

        auto output = engine.allocate_memory({ data_type, fmt, output_size});

        tile_ref<T>(input, output, output_axis, num_tiles);

        return output;
    }
};

using tile_test_f32 = tile_test<float>;
using tile_test_f16 = tile_test<half_t>;

TEST_P(tile_test_f32, test_case){
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(tile_test_f16, test_case) {
    ASSERT_NO_FATAL_FAILURE(test());
}

INSTANTIATE_TEST_SUITE_P(tile_gpu_2D,
                         tile_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateTileParams2D<float>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(layouts_2d)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(tile_gpu_2D,
                         tile_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateTileParams2D<half_t>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(layouts_2d)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(tile_gpu_3D,
                         tile_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateTileParams3D<float>()),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(layouts_3d)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(tile_gpu_3D,
                         tile_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateTileParams3D<half_t>()),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(layouts_3d)),
                         PrintToStringParamName());
