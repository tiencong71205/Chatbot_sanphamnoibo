import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 25,
  duration: '3m',
  gracefulStop: '2m',
  thresholds: {
    http_req_failed: ['rate<0.05'],
  },
};

const BASE_URL = 'http://host.docker.internal:8000';

const questions = [
  'Cảm biến cửa Mesh dùng để làm gì?',
  'Hướng dẫn sử dụng cảm biến cửa Mesh.',
  'Cảm biến cửa Mesh có tính năng gì?',
  'Thiết bị báo trạng thái kết nối như thế nào?',
];

export default function () {
  const question = questions[Math.floor(Math.random() * questions.length)];

  const response = http.post(
    `${BASE_URL}/api/chat`,
    JSON.stringify({ question: question }),
    {
      headers: { 'Content-Type': 'application/json' },
      timeout: '90s',
    }
  );

  check(response, {
    'status 200': (r) => r.status === 200,
    'co noi dung': (r) => r.body && r.body.length > 0,
  });

  sleep(1);
}
