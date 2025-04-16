#include <torch/library.h>

#include "../utils.h"
#include "bucket_fps/wrapper.h"

using torch::Tensor;

///////////////
//           //
//    CPU    //
//           //
///////////////

std::tuple<Tensor, Tensor> sample_cpu(const Tensor &x, int64_t k,
                                      torch::optional<int64_t> h,
                                      torch::optional<int64_t> start_idx) {
    TORCH_CHECK(x.device().is_cpu(), "x must be a CPU tensor, but found on ",
                x.device());
    TORCH_CHECK(x.dim() >= 2,
                "x must have at least 2 dims, but got size: ", x.sizes());
    TORCH_CHECK(k >= 1, "k must be greater than or equal to 1, but got ", k);
    auto [old_size, x_reshaped_raw] = bnorm_reshape(x);
    auto x_reshaped = x_reshaped_raw.to(torch::kFloat32).contiguous();

    auto height = h.value_or(5);

    torch::Tensor cur_start_idx;
    if (start_idx.has_value()) {
        cur_start_idx = torch::ones({x_reshaped.size(0)},
                                    x_reshaped.options().dtype(torch::kInt64)) *
                        start_idx.value();
    } else {
        cur_start_idx =
            torch::randint(0, x_reshaped.size(-2), {x_reshaped.size(0)},
                           x_reshaped.options().dtype(torch::kInt64));
    }

    Tensor ret_indices = torch::empty(
        {x_reshaped.size(0), k}, x_reshaped.options().dtype(torch::kInt64));

    if (x_reshaped.size(0) == 1) {
        // single batch
        bucket_fps_kdline(x_reshaped.const_data_ptr<float>(),
                          x_reshaped.size(1), x_reshaped.size(2), k,
                          cur_start_idx.const_data_ptr<int64_t>()[0], height,
                          ret_indices.mutable_data_ptr<int64_t>());
    } else {
        torch::parallel_for(
            0, x_reshaped.size(0), 0, [&](int64_t start, int64_t end) {
                for (auto i = start; i < end; i++) {
                    bucket_fps_kdline(
                        x_reshaped[i].const_data_ptr<float>(),
                        x_reshaped.size(1), x_reshaped.size(2), k,
                        cur_start_idx.const_data_ptr<int64_t>()[i], height,
                        ret_indices[i].mutable_data_ptr<int64_t>());
                }
            });
    }

    Tensor ret_tensor = torch::gather(
        x_reshaped_raw, 1,
        ret_indices.view({ret_indices.size(0), ret_indices.size(1), 1})
            .repeat({1, 1, x_reshaped.size(2)}));

    // reshape to original size
    auto ret_tensor_sizes = old_size.vec();
    ret_tensor_sizes[ret_tensor_sizes.size() - 2] = k;
    auto ret_indices_sizes = old_size.vec();
    ret_indices_sizes.pop_back();
    ret_indices_sizes[ret_indices_sizes.size() - 1] = k;

    return std::make_tuple(
        ret_tensor.view(ret_tensor_sizes),
        ret_indices.view(ret_indices_sizes).to(torch::kLong));
}

TORCH_LIBRARY_IMPL(torch_fpsample, CPU, m) { m.impl("sample", &sample_cpu); }
