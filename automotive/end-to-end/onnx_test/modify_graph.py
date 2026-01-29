import onnx
import onnx_graphsurgeon as gs

# Load and import the model
graph = gs.import_onnx(onnx.load("uniad_tiny_imgx0.25_cp.repaired.onnx"))

target_node = None
for node in graph.nodes:
    if node.op == 'InverseTRT':
        target_node = node
        print('found')
        break
# 1. Find the target node by name
print(target_node)
#target_node = [node for node in graph.nodes if node.name == "InverseTRT"][0]

# 2. Create the new custom "Inverse" node
# Use the same inputs and outputs as the original node to maintain connectivity
custom_node = gs.Node(
    op="Inverse",
    inputs=target_node.inputs, 
    outputs=target_node.outputs,
    name ="Inverse_custom",
    #attrs={"module": "ai.onnx.contrib"} # Custom domain is recommended
)

custom_node.domain = "ai.onnx.contrib"

for node in graph.nodes:
    for i, inp in enumerate(node.inputs):
        # If a node's input is the output of the old node, replace it with the new node's output
        if inp == target_node.outputs[0]:
            node.inputs[i] = custom_node.outputs[0]

# 3. Replace the node in the graph's node list
graph.nodes.remove(target_node)
graph.nodes.append(custom_node)
target_node.inputs = []
target_node.outputs = []
# 4. Clean up (removes dangling tensors) and save
graph.cleanup().toposort()
model = gs.export_onnx(graph)
model.ir_version = 8
custom_opset = model.opset_import.add()
custom_opset.domain = "ai.onnx.contrib"
custom_opset.version = 1 
onnx.save(model, "model_modified.onnx")