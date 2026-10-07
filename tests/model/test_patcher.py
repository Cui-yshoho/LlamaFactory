# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from llamafactory.model.patcher import _get_qwen3_5_language_layers


def test_get_qwen3_5_language_layers_supports_conditional_wrapper():
    expected_layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

    class _LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = expected_layers

    class _ConditionalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = _LanguageModel()

    assert _get_qwen3_5_language_layers(_ConditionalModel()) is expected_layers
