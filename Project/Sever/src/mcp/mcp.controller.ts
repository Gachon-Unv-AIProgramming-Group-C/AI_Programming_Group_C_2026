// McpController는 POST /mcp 엔드포인트를 담당한다
// Claude 또는 외부 클라이언트가 JSON-RPC 2.0 형식으로 요청을 보내면 이 컨트롤러가 받아서 McpService로 전달한다
import { Body, Controller, Post, HttpCode, HttpStatus, Logger } from '@nestjs/common';
import { McpService } from './mcp.service';
import { JsonRpcRequest } from './mcp.types';

@Controller('mcp')  // /mcp 경로에 대한 요청을 처리한다
export class McpController {
  private readonly logger = new Logger(McpController.name);

  // NestJS가 McpService를 자동으로 주입한다
  constructor(private readonly mcpService: McpService) {}

  @Post()
  @HttpCode(HttpStatus.OK)  // 항상 200 OK를 반환한다 (JSON-RPC 에러는 HTTP 에러가 아닌 응답 본문에 포함)
  async handleMcp(@Body() body: JsonRpcRequest | JsonRpcRequest[]) {
    // JSON-RPC 2.0 배치 요청: 배열로 들어오는 경우 각 요청을 병렬로 처리한다
    if (Array.isArray(body)) {
      this.logger.debug(`배치 요청 ${body.length}개`);
      const responsePromises = body.map((req) => this.mcpService.handleRequest(req));
      const responses = (await Promise.all(responsePromises))
        .filter((res): res is NonNullable<typeof res> => res !== null);  // null(알림 응답)은 제외
      return responses;
    }

    // 단일 JSON-RPC 요청 처리
    const result = await this.mcpService.handleRequest(body);

    // notifications(알림)는 응답이 필요 없으므로 null이 반환된다. 빈 객체로 응답한다
    if (result === null) return {};

    return result;
  }
}
