# frozen_string_literal: true

# source "https://rubygems.org"

# gemspec

# gem "html-proofer", "~> 5.0", group: :test

# platforms :mingw, :x64_mingw, :mswin, :jruby do
#   gem "tzinfo", ">= 1", "< 3"
#   gem "tzinfo-data"
# end

# gem "wdm", "~> 0.2.0", :platforms => [:mingw, :x64_mingw, :mswin]

# gem 'jekyll-seo-tag'

# gem "jekyll-sitemap"

# frozen_string_literal: true

source "https://rubygems.org"

# 지정된 Ruby 버전과 호환되도록 Jekyll 버전 명시
gem "jekyll", "~> 4.3"

# Jekyll 테마 (gemspec이 있으므로 현재 디렉터리를 가리킴)
gem "jekyll-theme-chirpy", "~> 6.0"

# 플러그인
group :jekyll_plugins do
  gem "jekyll-seo-tag"
  gem "jekyll-sitemap"
  gem "jekyll-archives"
  gem "jekyll-paginate"
end

# Windows와 JRuby는 특별한 gem이 필요
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Windows에 필요한 파일 시스템 알림 지원
gem "wdm", "~> 0.2.0", :platforms => [:mingw, :x64_mingw, :mswin]

# HTTP 테스트용
gem "html-proofer", "~> 5.0", group: :test

# Ruby 3.0 이상에 필요
gem "webrick", "~> 1.8"

# Gemfile에 추가
gem 'sass-embedded', '~> 1.64.1'  # 이전 안정 버전  