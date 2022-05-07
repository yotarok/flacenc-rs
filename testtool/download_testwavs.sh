#!/bin/sh
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

mkdir testwav

curl "https://upload.wikimedia.org/wikipedia/commons/9/97/%22I_Love_You%2C_California%22%2C_performed_by_the_Prince%27s_Orchestra_in_1914_for_Columbia_Records.oga" \
  | ffmpeg -i - testwav/wikimedia.i_love_you_california.wav

curl "https://upload.wikimedia.org/wikipedia/commons/b/bd/Drozerix_-_A_Winter_Kiss.wav" \
  > testwav/wikimedia.winter_kiss.wav

curl "https://upload.wikimedia.org/wikipedia/commons/7/7f/Jazz_Funk_no1_%28saxophone%29.flac" \
  | ffmpeg -i - testwav/wikimedia.jazz_funk_no1_sax.wav
