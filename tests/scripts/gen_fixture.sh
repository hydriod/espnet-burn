python ./tests/scripts/gen_fixture.py -n vits/generator/text_encoder/encoder/layer/attention -p tts.generator.text_encoder.encoder.encoders.0.self_attn. -i '{"query": "1,32,192","key": "1,32,192","value": "1,32,192","pos_emb": "1,63,192","mask": "1,1,32"}'
python ./tests/scripts/gen_fixture.py -n vits/generator/text_encoder/encoder/layer/multi_layered_conv -p tts.generator.text_encoder.encoder.encoders.0.feed_forward. -i '{"x":"1,32,192"}'
python ./tests/scripts/gen_fixture.py -n vits/generator/text_encoder/encoder/layer/mod -p tts.generator.text_encoder.encoder.encoders.0. -i '{"x_input":["1,32,192", "1,63,192"], "mask":"1,1,32"}'
python ./tests/scripts/gen_fixture.py -n vits/generator/text_encoder/encoder/positional_encoding -p tts.generator.text_encoder.encoder.embed.0. -i '{"x": "1,32,192"}'
python ./tests/scripts/gen_fixture.py -n vits/generator/text_encoder/encoder/mod -p tts.generator.text_encoder.encoder. -i '{"xs": "1,32,192", "masks": "1,1,32"}'
python ./tests/scripts/gen_fixture.py -n vits/generator/text_encoder/mod -p tts.generator.text_encoder. -i '{"x": "1,32", "x_lengths": "1"}' --input-type '{"x": "randint(0,47)", "x_lengths": "randint(32,33)"}'
python ./tests/scripts/gen_fixture.py -n vits/generator/duration_predictor/mod -p tts.generator.duration_predictor. -i '{"x": "1,192,32", "x_mask": "1,1,32", "inverse": "1", "noise_scale": "1"}' --input-type '{"inverse": "randint(1,2)", "noise_scale":"randint(0,1)"}'
python ./tests/scripts/gen_fixture.py -n vits/generator/duration_predictor/flow_step/conv/dds -p tts.generator.duration_predictor.dds. -i '{"x": "1,192,32", "x_mask": "1,1,32"}'
python ./tests/scripts/gen_fixture.py -n vits/generator/duration_predictor/flow_step/conv/mod -p tts.generator.duration_predictor.flows.1. -i '{"x": "1,2,32", "x_mask": "1,1,32", "inverse": "1"}' --input-type '{"inverse": "randint(1,2)"}'
python ./tests/scripts/gen_fixture.py -n modules/wavenet -p tts.generator.flow.flows.0.encoder. -i '{"x": "1,192,32", "x_mask": "1,1,32"}'
python ./tests/scripts/gen_fixture.py -n modules/hifi_gan/generator -p tts.generator.decoder. -i '{"c": "1,192,8"}'
python ./tests/scripts/gen_fixture.py -n modules/residual_coupling -p tts.generator.flow. -i '{"x": "1,192,32", "x_mask": "1,1,32", "inverse": "1"}' --input-type '{"inverse": "randint(1,2)"}'
python ./tests/scripts/gen_fixture.py -n vits/generator/mod -p tts.generator. -i '{"text": "1,8", "text_lengths": "1", "noise_scale": "1", "noise_scale_dur": "1"}' --input-type '{"text": "randint(0,47)", "text_lengths": "randint(8,9)", "noise_scale": "randint(0,1)", "noise_scale_dur": "randint(0,1)"}' -f inference
python ./tests/scripts/tts2speech.py
