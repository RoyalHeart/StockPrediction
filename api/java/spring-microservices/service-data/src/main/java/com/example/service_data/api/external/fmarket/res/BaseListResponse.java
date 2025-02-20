package com.example.service_data.api.external.fmarket.res;

import java.util.List;

import lombok.Data;

@Data
public class BaseListResponse<D, E> {
    private String status;
    private String code;
    private String message;
    private List<D> data;
    private E extra;
}
