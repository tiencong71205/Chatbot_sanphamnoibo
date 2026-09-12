import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 10,
  duration: '30s',
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
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: '90s',
    }
  );

  if (r.status !== 200 || r.error) {
    console.error(JSON.stringify({
      status: r.status,
      error: r.error,
      error_code: r.error_code,
      duration_ms: r.timings.duration,
      blocked_ms: r.timings.blocked,
      connecting_ms: r.timings.connecting,
      waiting_ms: r.timings.waiting,
      receiving_ms: r.timings.receiving,
    }));
  }

  check(r, {
    'status 200': (res) => res.status === 200,
    'co noi dung': (res) => res.body && res.body.length > 0,
  });

  sleep(1);
}
