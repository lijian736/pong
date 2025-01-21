import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import onnx
import onnxruntime
import numpy as np


class Policy(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.layer1 = nn.Linear(5, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 32)
        self.layer4 = nn.Linear(32, action_size)

    def forward(self, input):
        x = F.relu(self.layer1(input))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        x = torch.argmax(x, dim=1)
        return x


if __name__ == "__main__":
    # Step 1. the pytorch model weight path
    pytorch_model_weight_path = "./reinforce_model_params.pth"
    # the action space number
    action_space_n = 2

    # Step 2. load the pytorch weights
    policy_model = Policy(action_space_n)
    policy_model.load_state_dict(
        torch.load(
            pytorch_model_weight_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    )
    policy_model.eval()

    # Step 3. define the input(batch size is 1, and input size is 5)
    dummy_input = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
    print(f"dummy_input: {dummy_input.shape}, {dummy_input.dtype}")

    # Step 4. export the ONNX model
    torch.onnx.export(
        policy_model,
        dummy_input,
        "./reinforce_model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}},
    )

    print(f"The model has been exported to .ONNX file successfully.")

    # Step 4. verify the output .ONNX file
    onnx_policy_model = onnx.load("./reinforce_model.onnx")
    onnx.checker.check_model(onnx_policy_model)
    print(f"The saved .ONNX model has been checked")

    # create the onnx runtime session
    ort_session = onnxruntime.InferenceSession("./reinforce_model.onnx")

    ort_input = ort_session.get_inputs()[0]
    ort_input_name = ort_input.name
    ort_input_shape = ort_input.shape
    ort_input_type = ort_input.type
    print(
        f"The onnx runtime input: {ort_input_name}, {ort_input_shape}, {ort_input_type}"
    )

    ort_output = ort_session.get_outputs()[0]
    ort_output_name = ort_output.name
    ort_output_shape = ort_output.shape
    ort_output_type = ort_output.type
    print(
        f"The onnx runtime output: {ort_output_name}, {ort_output_shape}, {ort_output_type}"
    )

    x = np.random.rand(100, 5).astype(np.float32)
    ort_result = ort_session.run([ort_output_name], {ort_input_name: x})[0]
    ort_result = ort_result.flatten()

    x = torch.from_numpy(x)
    torch_output = policy_model(x).detach().numpy().flatten()

    diff_max = np.max(ort_result - torch_output)
    diff_min = np.min(ort_result - torch_output)
    print(f"The difference between ONNX runtime and PyTorch: {diff_max}, {diff_min}")
