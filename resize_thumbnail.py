import os
import argparse
from PIL import Image
# import cairosvg  # SVG를 PNG로 변환하기 위한 라이브러리

def reshape_image(input_path, output_path, max_size=(1200, 628), target_ratio=1.91, bg_color=(255, 255, 255)):
    """
    이미지를 잘리지 않게 1.91:1 비율로 조정하고, 필요시 1200×628 이하로 리사이징하는 함수
    SVG 파일도 지원합니다.
    
    Args:
        input_path (str): 입력 이미지 경로
        output_path (str): 출력 이미지 경로
        max_size (tuple): 최대 이미지 크기 (너비, 높이)
        target_ratio (float): 목표 가로세로 비율 (너비/높이)
        bg_color (tuple): 배경 색상 (R, G, B)
    """
    try:
        # SVG 파일인지 확인
        is_svg = input_path.lower().endswith('.svg')
        
        # SVG 파일이면 먼저 PNG로 변환
        if is_svg:
            # 출력 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 임시 PNG 파일 경로 생성
            temp_png_path = os.path.splitext(output_path)[0] + "_temp.png"
            
            # SVG를 PNG로 변환 (기본 크기로)
            # 여기서는 최대 크기의 4배 정도의 큰 크기로 변환하여 품질 손실을 최소화
            png_size = max(max_size[0] * 2, max_size[1] * 2)
            cairosvg.svg2png(url=input_path, write_to=temp_png_path, 
                            output_width=png_size, output_height=png_size)
            
            # 생성된 PNG 파일 열기
            img = Image.open(temp_png_path)
        else:
            # 일반 이미지 파일 열기
            img = Image.open(input_path)
        
        # 원본 이미지 크기
        original_width, original_height = img.size
        original_ratio = original_width / original_height
        
        # 비율을 유지하면서 대상 비율의 캔버스에 맞게 리사이징
        if original_ratio > target_ratio:
            # 이미지가 더 가로로 넓은 경우
            # 너비를 기준으로 높이 계산
            new_width = min(original_width, max_size[0])
            new_height = int(new_width / target_ratio)
            
            # 이미지 비율은 유지하면서 리사이징
            img_width = new_width
            img_height = int(img_width / original_ratio)
            
            # 이미지가 새 캔버스보다 크면 비율 유지하며 축소
            if img_height > new_height:
                img_height = new_height
                img_width = int(img_height * original_ratio)
        else:
            # 이미지가 더 세로로 높은 경우
            # 높이를 기준으로 너비 계산
            new_height = min(original_height, max_size[1])
            new_width = int(new_height * target_ratio)
            
            # 이미지 비율은 유지하면서 리사이징
            img_height = new_height
            img_width = int(img_height * original_ratio)
            
            # 이미지가 새 캔버스보다 크면 비율 유지하며 축소
            if img_width > new_width:
                img_width = new_width
                img_height = int(img_width / original_ratio)
        
        # 최대 크기 제한 적용
        if new_width > max_size[0]:
            new_width = max_size[0]
            new_height = int(new_width / target_ratio)
        
        if new_height > max_size[1]:
            new_height = max_size[1]
            new_width = int(new_height * target_ratio)
        
        # 이미지 리사이징
        resized_img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
        
        # 새 캔버스 생성 (배경색 지정)
        new_img = Image.new('RGB', (new_width, new_height), bg_color)
        
        # 이미지를 캔버스 중앙에 배치
        paste_x = (new_width - img_width) // 2
        paste_y = (new_height - img_height) // 2
        
        # 투명도가 있는 이미지인 경우를 처리
        if resized_img.mode == 'RGBA':
            # 알파 채널 분리
            r, g, b, a = resized_img.split()
            resized_rgb = Image.merge('RGB', (r, g, b))
            
            new_img.paste(resized_rgb, (paste_x, paste_y), mask=a)
        else:
            new_img.paste(resized_img, (paste_x, paste_y))
        
        # 출력 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 이미지 저장 (SVG는 PNG로 변환, 다른 포맷은 원본 유지)
        if is_svg:
            # SVG 파일은 PNG로 저장
            output_path = os.path.splitext(output_path)[0] + ".png"
            new_img.save(output_path, format='PNG', quality=95)
            
            # 임시 파일 삭제
            if os.path.exists(temp_png_path):
                os.remove(temp_png_path)
        else:
            # 원본 포맷 유지를 위한 설정
            original_format = img.format if img.format else 'JPEG'
            
            # JPEG 포맷인 경우 quality 설정
            if original_format == 'JPEG':
                new_img.save(output_path, format=original_format, quality=95)
            else:
                new_img.save(output_path, format=original_format)
        
        # 크기 정보를 포함한 메시지 출력
        if is_svg:
            print(f"SVG 이미지 변환 완료: {input_path} → {output_path} ({new_width}x{new_height})")
        else:
            print(f"이미지 변환 완료: {input_path} ({original_width}x{original_height}) → {output_path} ({new_width}x{new_height})")
            
        # 메모리에서 이미지 해제
        img.close()
        new_img.close()
        
    except Exception as e:
        print(f"오류 발생: {input_path} 처리 중 - {str(e)}")

def process_directory(input_dir, output_dir, max_size=(1200, 628), target_ratio=1.91, bg_color=(255, 255, 255)):
    """
    디렉토리 내의 모든 이미지를 1.91:1 비율로 조정하고 필요시 리사이징하는 함수
    
    Args:
        input_dir (str): 입력 이미지 디렉토리
        output_dir (str): 출력 이미지 디렉토리
        max_size (tuple): 최대 이미지 크기 (너비, 높이)
        target_ratio (float): 목표 가로세로 비율 (너비/높이)
        bg_color (tuple): 배경 색상 (R, G, B)
    """
    # 지원하는 이미지 포맷 확장자 (SVG 추가)
    supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'}
    
    # 입력 디렉토리가 존재하는지 확인
    if not os.path.exists(input_dir):
        print(f"오류: 입력 디렉토리가 존재하지 않습니다 - {input_dir}")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 처리된 파일 수 카운터
    processed_count = 0
    
    # 모든 파일 처리
    for root, _, files in os.walk(input_dir):
        for file in files:
            # 파일 확장자 확인
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in supported_formats:
                # 입력 및 출력 경로 설정
                rel_path = os.path.relpath(root, input_dir)
                input_path = os.path.join(root, file)
                
                # 상대 경로가 '.'인 경우 빈 문자열로 변경
                if rel_path == '.':
                    rel_path = ''
                
                # SVG 파일은 PNG로 저장될 수 있도록 출력 경로 설정
                if file_ext == '.svg':
                    file_without_ext = os.path.splitext(file)[0]
                    output_path = os.path.join(output_dir, rel_path, file_without_ext + ".png")
                else:
                    output_path = os.path.join(output_dir, rel_path, file)
                
                # 이미지 리사이징
                reshape_image(input_path, output_path, max_size, target_ratio, bg_color)
                processed_count += 1
    
    print(f"총 {processed_count}개 파일 처리 완료")

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='이미지를 잘리지 않게 1.91:1 비율로 조정하고 필요시 1200×628 픽셀 이하로 조정합니다. SVG 파일도 지원합니다.')
    parser.add_argument('--input', default='./assets/img/posts/resize/input', help='입력 이미지 파일 또는 디렉토리')
    parser.add_argument('--output', default='./assets/img/posts/resize/output', help='출력 이미지 파일 또는 디렉토리')
    parser.add_argument('--max-width', type=int, default=1200, help='최대 출력 이미지 너비 (기본값: 1200)')
    parser.add_argument('--max-height', type=int, default=628, help='최대 출력 이미지 높이 (기본값: 628)')
    parser.add_argument('--ratio', type=float, default=1.91, help='목표 가로세로 비율 (기본값: 1.91)')
    parser.add_argument('--bg-color', nargs=3, type=int, default=[255, 255, 255], 
                      help='배경 색상 RGB 값 (기본값: 255 255 255 - 흰색)')
    
    args = parser.parse_args()
    
    # 최대 크기 설정 
    max_size = (args.max_width, args.max_height)
    
    # 배경색 설정
    bg_color = tuple(args.bg_color)
    
    # 입력이 디렉토리인 경우
    if os.path.isdir(args.input):
        process_directory(args.input, args.output, max_size, args.ratio, bg_color)
    # 입력이 파일인 경우
    elif os.path.isfile(args.input):
        # SVG 파일인 경우 출력 경로 조정
        if args.input.lower().endswith('.svg'):
            # SVG는 PNG로 저장되므로 확장자 변경
            output_path = os.path.splitext(args.output)[0] + ".png"
            reshape_image(args.input, output_path, max_size, args.ratio, bg_color)
        else:
            reshape_image(args.input, args.output, max_size, args.ratio, bg_color)
    else:
        print(f"오류: 입력 파일 또는 디렉토리가 존재하지 않습니다 - {args.input}")

if __name__ == "__main__":
    main()