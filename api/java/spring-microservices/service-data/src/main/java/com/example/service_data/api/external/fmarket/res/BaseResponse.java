package com.example.service_data.api.external.fmarket.res;

import lombok.Data;

@Data
public class BaseResponse<D, E> {
    private String status;
    private String code;
    private String message;
    private D data;
    private E extra;
}
