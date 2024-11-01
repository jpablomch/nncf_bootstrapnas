# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Tuple

from nncf import NNCFConfig
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_builder import ElasticityBuilder
from nncf.experimental.torch.nas.bootstrapNAS.training.scheduler import NASSchedulerParams
from nncf.experimental.torch.sqft.training.neural_lora_search_controller import NeuralLoraSearchController
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.torch.algo_selector import ZeroCompressionLoss
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.nncf_network import NNCFNetwork


class NLSBuilderStateNames:
    ELASTICITY_BUILDER_STATE = "elasticity_builder_state"


@PT_COMPRESSION_ALGORITHMS.register("neural_lora_search")
class NeuralLoraSearchBuilder(PTCompressionAlgorithmBuilder):
    """
    Determines which modifications should be made to the original FP32 model in
    order to train a supernet using Progressive Shrinking procedure from OFA (https://arxiv.org/abs/1908.09791).
    Operates on an NNCFNetwork object wrapping a target PyTorch model (torch.nn.Module).
    """

    _state_names = NLSBuilderStateNames

    def __init__(self, nncf_config: NNCFConfig, should_init: bool = True):
        super().__init__(nncf_config, should_init)
        self._elasticity_builder = ElasticityBuilder(self.config, self.should_init)
        self._lr_schedule_config = self._algo_config.get("lr_schedule", {})

    def initialize(self, model: NNCFNetwork) -> None:
        """
        Initialize model parameters before training

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        """

    def _get_algo_specific_config_section(self) -> Dict:
        return self.config.get("SQFT", {}).get("training", {})

    def _build_controller(self, model: NNCFNetwork) -> "NeuralLoraSearchController":
        elasticity_ctrl = self._elasticity_builder.build_controller(model)
        schedule_params = NASSchedulerParams.from_config(self._algo_config.get("schedule", {}))
        return NeuralLoraSearchController(
            model,
            elasticity_ctrl,
            schedule_params,
            self._lr_schedule_config,
            ZeroCompressionLoss,
        )

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        return self._elasticity_builder.get_transformation_layout(target_model)

    def _get_state_without_name(self) -> Dict[str, Any]:
        """
        Implementation of get_state that returns state without builder name.

        :return: Returns a dictionary with Python data structures
            (dict, list, tuple, str, int, float, True, False, None) that represents state of the object.
        """
        return {
            self._state_names.ELASTICITY_BUILDER_STATE: self._elasticity_builder.get_state(),
        }

    def _load_state_without_name(self, state_without_name: Dict[str, Any]):
        """
        Implementation of load state that takes state without builder name.

        :param state_without_name: Output of `_get_state_without_name()` method.
        """
        elasticity_builder_state = state_without_name[self._state_names.ELASTICITY_BUILDER_STATE]
        self._elasticity_builder.load_state(elasticity_builder_state)

    def _are_frozen_layers_allowed(self) -> Tuple[bool, str]:
        """
        Frozen layers will be allowed in Neural Lora Search algorithm.
        It freezes the pretrained weights while training the LoRA Super-Adapter.

        :return: A tuple where the first element is a boolean indicating if frozen layers are allowed,
                 and the second element is a string message explaining the reason.
        """
        return True, "Frozen layers are allowed under the `Neural Lora Search` algorithm"
