from model_test import ModelTest


class TestQwen2_5(ModelTest):
    # NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct"
    NATIVE_MODEL_ID = "/root/projects/fanfiction-go/python/ai/train/config_file/config_qwen_qwq_32b_preview.ini"
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.2
    NATIVE_ARC_CHALLENGE_ACC = 0.2739
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3055
    TRUST_REMOTE_CODE = False
    APPLY_CHAT_TEMPLATE = True
    BATCH_SIZE = 6

    def test_qwen2_5(self):
        self.quant_lm_eval()
