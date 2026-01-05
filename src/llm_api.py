from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# 从环境变量获取模型路径
model_path = os.environ.get('MODEL_PATH', '/app/models/model.rkllm')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_path': model_path})

@app.route('/infer', methods=['POST'])
def infer():
    try:
        data = request.json
        prompt = data.get('prompt', '')

        # 这里应该集成RKLLM推理代码
        # 由于实际的RKLLM推理代码未提供，这里使用模拟实现
        # 在实际部署中，这里应该调用RKLLM推理接口
        response = f"Model response for: {prompt}"

        return jsonify({
            'response': response,
            'model_path': model_path,
            'model_type': os.path.basename(model_path) if model_path else 'unknown'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)