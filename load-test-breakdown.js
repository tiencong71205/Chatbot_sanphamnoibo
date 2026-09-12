import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';

const backendLatency = new Trend('backend_latency_ms');
const retrievalLatency = new Trend('retrieval_latency_ms');
const generationLatency = new Trend('generation_latency_ms');

export const options = {
  vus: 10,
  duration: '1m',
};

const BASE_URL = 'http://host.docker.internal:8000';

const questions = [
  'Cảm biến cửa Mesh dùng để làm gì?',
  'Hướng dẫn sử dụng cảm biến cửa Mesh.',
  'Cảm biến cửa Mesh có tính năng gì?',
  'Thiết bị báo trạng thái kết nối như thế nào?',
];

export default function () {
  const question =
    questions[Math.floor(Math.random() * questions.length)];

  const r = http.post(
    `${BASE_URL}/api/chat`,
    JSON.stringify({ question }),
    {
      headers: { 'Content-Type': 'application/json' },
      timeout: '90s',
    }
  );

  check(r, {
    'status 200': (res) => res.status === 200,
    'co noi dung': (res) => res.body && res.body.length > 0,
  });

  if (r.status === 200) {
    try {
      const data = JSON.parse(r.body);

      if (data.latency_ms != null)
        backendLatency.add(data.latency_ms);

      if (data.retrieval_ms != null)
        retrievalLatency.add(data.retrieval_ms);

      if (data.generation_ms != null)
        generationLatency.add(data.generation_ms);

    } catch (e) {
      console.error(`JSON parse error: ${e}`);
    }
  }

  sleep(1);
}
