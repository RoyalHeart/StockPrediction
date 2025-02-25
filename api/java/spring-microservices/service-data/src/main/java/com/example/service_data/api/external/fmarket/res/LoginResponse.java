package com.example.service_data.api.external.fmarket.res;

import com.example.service_data.api.external.fmarket.res.LoginResponse.LoginData;

import lombok.Data;

public class LoginResponse extends FmBaseResponse<LoginData, LoginData> {

    @Data
    public class LoginData {
        private String accessToken;
    }
}
