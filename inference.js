const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const ort = require('onnxruntime-node');

// ==================== 설정 (Python 코드와 동기화) ====================

const MODEL_CONFIGS = {
    "melon": { width: 230, height: 70, modelFile: "model_melon.onnx" },
    "nol": { width: 210, height: 70, modelFile: "model_nol.onnx" }
};

// 알파벳 대문자 (A-Z) 매핑
const ALPHABETS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const IDX_TO_CHAR = {};
for (let i = 0; i < ALPHABETS.length; i++) {
    IDX_TO_CHAR[i] = ALPHABETS[i];
}
const BLANK_LABEL = 26; // 0~25: A~Z, 26: Blank

// ==================== 유틸리티 함수 ====================

/**
 * 이미지 전처리 함수
 * - 이미지를 읽어서 Grayscale 변환
 * - 모델 크기에 맞게 Resize (Linear/Bilinear 보간)
 * - 0~1 사이 값으로 정규화 (Normalization)
 * - Float32Array 텐서 데이터로 변환 (NCHW 포맷: 1x1xHxW)
 */
async function preprocessImage(imagePath, config) {
    try {
        // 1. 이미지 로드 및 변환
        const { data, info } = await sharp(imagePath)
            .resize(config.width, config.height, { 
                fit: 'fill',       // 비율 무시하고 강제 리사이즈 (파이썬 로직과 동일)
                kernel: 'linear'   // Python의 Image.BILINEAR와 유사
            })
            .grayscale()           // 'L' 모드 변환
            .raw()                 // 픽셀 데이터 추출
            .toBuffer({ resolveWithObject: true });

        // 2. 정규화 및 Tensor 변환 (Uint8 -> Float32, 0~255 -> 0.0~1.0)
        const float32Data = new Float32Array(data.length);
        for (let i = 0; i < data.length; i++) {
            float32Data[i] = data[i] / 255.0;
        }

        // 3. ONNX Runtime용 Tensor 생성 (Dims: [Batch=1, Channel=1, Height, Width])
        const tensor = new ort.Tensor('float32', float32Data, [1, 1, config.height, config.width]);
        return tensor;

    } catch (e) {
        throw new Error(`이미지 전처리 실패: ${e.message}`);
    }
}

/**
 * CTC Decoding 함수 (Greedy Search)
 * - Logits(Output)에서 가장 높은 확률의 인덱스 추출
 * - 중복된 문자 제거 및 Blank 라벨 제거
 */
function ctcDecode(outputTensor) {
    // outputTensor 구조: [seq_len, batch_size, num_classes] 
    // 파이썬 모델 출력: (Width/Seq, Batch, Class) -> 예: [Width크기, 1, 27]
    
    const dims = outputTensor.dims; // [seq_len, batch, num_classes]
    const seqLen = dims[0];
    const numClasses = dims[2]; // 27
    const data = outputTensor.data;

    let predictedText = "";
    let prevIndex = -1;

    // 시퀀스(Time Step) 순회
    for (let t = 0; t < seqLen; t++) {
        // 현재 Time Step (t)에서의 ArgMax 찾기
        let maxVal = -Infinity;
        let maxIdx = -1;
        
        // 현재 step의 시작 오프셋
        const offset = t * numClasses;

        for (let c = 0; c < numClasses; c++) {
            if (data[offset + c] > maxVal) {
                maxVal = data[offset + c];
                maxIdx = c;
            }
        }

        // CTC 로직: 이전 문자와 다르고, Blank가 아니면 추가
        if (maxIdx !== prevIndex && maxIdx !== BLANK_LABEL) {
            if (IDX_TO_CHAR[maxIdx]) {
                predictedText += IDX_TO_CHAR[maxIdx];
            }
        }
        prevIndex = maxIdx;
    }

    return predictedText;
}

// ==================== 메인 추론 함수 ====================

async function runInference(imagePath, modelType) {
    console.time("Inference Time");
    
    // 1. 설정 로드
    const config = MODEL_CONFIGS[modelType];
    if (!config) throw new Error(`지원하지 않는 모델 타입입니다: ${modelType}`);

    const modelPath = path.join(__dirname, 'models', config.modelFile);
    if (!fs.existsSync(modelPath)) throw new Error(`모델 파일을 찾을 수 없습니다: ${modelPath}`);

    try {
        // 2. 세션 생성 (모델 로드)
        const session = await ort.InferenceSession.create(modelPath);

        // 3. 이미지 전처리
        const inputTensor = await preprocessImage(imagePath, config);

        // 4. 추론 실행
        // 'input'은 ONNX export 시 지정한 input_names와 일치해야 함
        const feeds = { input: inputTensor };
        const results = await session.run(feeds);

        // 5. 결과 디코딩
        // 'output'은 ONNX export 시 지정한 output_names와 일치해야 함
        const outputTensor = results.output;
        const text = ctcDecode(outputTensor);

        console.timeEnd("Inference Time");
        return text;

    } catch (e) {
        console.error("추론 중 오류 발생:", e);
        throw e;
    }
}

// ==================== 실행 예제 ====================

// 실제 사용 시에는 이 부분 호출
(async () => {
    // melon
    // 테스트용 설정 (경로는 실제 환경에 맞게 수정 필요)
    const melonImg = "melon.png"; 
    const melon = "melon"; // or "nol"

    console.log(`🚀 추론 시작 (Type: ${melon}, Img: ${melonImg})`);
    
    try {
        const result = await runInference(melonImg, melon);
        console.log("------------------------------------------------");
        console.log(`📝 예측 결과: ${result}`);
        console.log("------------------------------------------------");
    } catch (error) {
        console.error("❌ 실패:", error.message);
    }    
    
    const nolImg = "nol.png"; 
    const nol = "nol"; // or "nol"

    console.log(`🚀 추론 시작 (Type: ${nol}, Img: ${nolImg})`);
    
    try {
        const result = await runInference(nolImg, nol);
        console.log("------------------------------------------------");
        console.log(`📝 예측 결과: ${result}`);
        console.log("------------------------------------------------");
    } catch (error) {
        console.error("❌ 실패:", error.message);
    }
})();

