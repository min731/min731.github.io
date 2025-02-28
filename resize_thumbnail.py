import os
import argparse
from PIL import Image

def resize_image(input_path, output_path, target_size=(1200, 628)):
    """
    이미지를 지정된 크기로 리사이징하는 함수
    
    Args:
        input_path (str): 입력 이미지 경로
        output_path (str): 출력 이미지 경로
        target_size (tuple): 목표 이미지 크기 (너비, 높이)
    """
    try:
        # 이미지 열기
        with Image.open(input_path) as img:
            # 이미지 리사이징 (가로세로 비율 유지하지 않고 지정된 크기로 변환)
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # 출력 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 이미지 저장 (원본 이미지와 동일한 포맷 유지)
            resized_img.save(output_path)
            
            print(f"이미지 변환 완료: {input_path} → {output_path}")
    except Exception as e:
        print(f"오류 발생: {input_path} 처리 중 - {str(e)}")

def process_directory(input_dir, output_dir, target_size=(1200, 628)):
    """
    디렉토리 내의 모든 이미지를 리사이징하는 함수
    
    Args:
        input_dir (str): 입력 이미지 디렉토리
        output_dir (str): 출력 이미지 디렉토리
        target_size (tuple): 목표 이미지 크기 (너비, 높이)
    """
    # 지원하는 이미지 포맷 확장자
    supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    # 입력 디렉토리가 존재하는지 확인
    if not os.path.exists(input_dir):
        print(f"오류: 입력 디렉토리가 존재하지 않습니다 - {input_dir}")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
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
                
                output_path = os.path.join(output_dir, rel_path, file)
                
                # 이미지 리사이징
                resize_image(input_path, output_path, target_size)

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='이미지를 1200×628 픽셀 크기로 변환합니다.')
    parser.add_argument('--input', default='./assets/img/posts/resize/input' ,help='입력 이미지 파일 또는 디렉토리')
    parser.add_argument('--output', default='./assets/img/posts/resize/output',help='출력 이미지 파일 또는 디렉토리')
    parser.add_argument('--width', type=int, default=1200, help='출력 이미지 너비 (기본값: 1200)')
    parser.add_argument('--height', type=int, default=628, help='출력 이미지 높이 (기본값: 628)')
    
    args = parser.parse_args()
    
    # 목표 크기 설정
    target_size = (args.width, args.height)
    
    # 입력이 디렉토리인 경우
    if os.path.isdir(args.input):
        process_directory(args.input, args.output, target_size)
    # 입력이 파일인 경우
    elif os.path.isfile(args.input):
        resize_image(args.input, args.output, target_size)
    else:
        print(f"오류: 입력 파일 또는 디렉토리가 존재하지 않습니다 - {args.input}")

if __name__ == "__main__":
    main()